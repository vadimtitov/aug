"""Typed client for the official MCP Registry API.

https://registry.modelcontextprotocol.io lists publicly published MCP servers.
Parsing is deliberately tolerant: the registry is a young, evolving public API,
and a field it renames or drops should degrade one listing rather than crash
the whole search — see ``_parse_server``.

Only stdio (npm/pypi packages run via npx/uvx) and remote HTTP servers are
represented. Docker-based packages are skipped — see the MCP PRD, "Out of
scope" (v1 has no Docker socket access).
"""

import logging
import re
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from aug.utils.file_settings import McpServerConfig

logger = logging.getLogger(__name__)

_REGISTRY_BASE = "https://registry.modelcontextprotocol.io"
_TIMEOUT = 15.0
_DEFAULT_LIMIT = 10
# A literal/default value the registry declares as e.g. "{other_var}" is a
# template referencing some other input this v1 install flow has no way to
# resolve — see _split_inputs.
_TEMPLATE_RE = re.compile(r"\{[^{}]+\}")
_SLUG_INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")


class McpRegistryServer(BaseModel):
    """One search result, reduced to what installing it needs.

    ``required_inputs`` names the runtime inputs the server needs — env var
    names for stdio, HTTP header names for http. These are the literal names
    the server reads; they are never themselves hushed secret names (a header
    like "X-API-Key" isn't a valid one) — see ``to_config``'s ``bindings``
    parameter for how a caller maps each to an actual secret.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    namespace: str
    description: str
    version: str
    transport: Literal["stdio", "http"]
    command: str = ""  # stdio only
    args: list[str] = []  # stdio only — includes the pinned version + literal arguments
    url: str = ""  # http only
    required_inputs: list[str] = []
    # Env vars / headers the registry declares with a concrete, non-secret
    # value or default — these need no hushed binding, but the value itself
    # must still reach the installed server (see _split_inputs).
    literal_inputs: dict[str, str] = {}

    @property
    def slug(self) -> str:
        """Config + tool-namespace-safe short name, disambiguated by publisher.

        Two different registry namespaces can legitimately publish an
        identically-named package (e.g. two accounts both shipping
        "server-postgres") — a slug built from the tail alone would silently
        collide, and installing the second would read as "already
        configured" instead of the different server it actually is. The
        namespace's last dot-segment (the publishing account, e.g. "acme" in
        "io.github.acme") is what actually distinguishes them, so it's
        folded in; the registry's generic domain prefix ("io.github.") is
        not, since it says nothing about who published it.
        """
        tail = self.name.rsplit("/", 1)[-1]
        tail = tail.removeprefix("server-") if tail.startswith("server-") else tail
        account = self.namespace.rsplit(".", 1)[-1].lower()
        account = _SLUG_INVALID_CHARS_RE.sub("-", account).strip("-")
        if not account or tail == account or tail.startswith(f"{account}-"):
            return tail
        return f"{account}-{tail}"

    def to_config(self, bindings: dict[str, str]) -> McpServerConfig:
        """Build the settings.json entry for this server.

        ``bindings`` maps each entry in ``required_inputs`` (an env var or
        header name) to its ``hushed:KEY`` reference — built by the caller
        once it has decided which secret backs each input. ``literal_inputs``
        rides along separately since those are plain values, never secret
        references.
        """
        if self.transport == "stdio":
            return McpServerConfig(
                name=self.slug,
                transport="stdio",
                command=self.command,
                args=self.args,
                env=bindings,
                env_static=self.literal_inputs,
                enabled=True,
            )
        return McpServerConfig(
            name=self.slug,
            transport="http",
            url=self.url,
            headers=bindings,
            headers_static=self.literal_inputs,
            enabled=True,
        )


class McpRegistryClient:
    """Thin async client over the MCP Registry search endpoint."""

    def __init__(self, base_url: str = _REGISTRY_BASE) -> None:
        self._base = base_url.rstrip("/")

    async def search(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[McpRegistryServer]:
        """Search the registry. Returns [] on no matches; raises on transport failure."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{self._base}/v0.1/servers", params={"search": query, "limit": limit}
            )
            r.raise_for_status()
        data = r.json()

        servers: list[McpRegistryServer] = []
        for entry in data.get("servers", [])[:limit]:
            # The live API wraps each result as {"server": {...}, "_meta": {...}};
            # unwrapping defensively (falling back to the entry itself) keeps this
            # working if a future schema version flattens it again.
            raw = entry.get("server", entry) if isinstance(entry, dict) else entry
            parsed = _parse_server(raw)
            if parsed is not None:
                servers.append(parsed)
        return servers


def _parse_server(raw: dict) -> McpRegistryServer | None:
    """Best-effort parse of one registry entry. Returns None if it can't be installed.

    Prefers a stdio package (npm -> npx, pypi -> uvx) when present, otherwise
    falls back to the first remote HTTP entry. An entry with neither, one whose
    package registry type we don't recognize (e.g. "oci", "nuget" — no Docker
    socket access and no .NET runtime in this container), or one whose package
    itself listens over HTTP rather than stdio (needs port allocation we don't
    do in v1), is skipped rather than raising — one unsupported listing must
    never sink the whole search.
    """
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    namespace = name.rsplit("/", 1)[0] if "/" in name else name
    description = raw.get("description", "") or ""
    server_version = str(raw.get("version") or "")

    package = _pick_package(raw.get("packages") or [], server_version)
    if package is not None:
        command, args, required_inputs, literal_inputs = package
        return McpRegistryServer(
            name=name,
            namespace=namespace,
            description=description,
            version=server_version,
            transport="stdio",
            command=command,
            args=args,
            required_inputs=required_inputs,
            literal_inputs=literal_inputs,
        )

    remote = _pick_remote(raw.get("remotes") or [])
    if remote is not None:
        url, required_inputs, literal_inputs = remote
        return McpRegistryServer(
            name=name,
            namespace=namespace,
            description=description,
            version=server_version,
            transport="http",
            url=url,
            required_inputs=required_inputs,
            literal_inputs=literal_inputs,
        )

    logger.debug("mcp_registry: skipping %r — no installable package or remote", name)
    return None


def _pick_package(
    packages: list[dict], server_version: str
) -> tuple[str, list[str], list[str], dict[str, str]] | None:
    """Return (command, args, required_inputs, literal_inputs) for the first
    supported package.

    Only npm (-> npx) and pypi (-> uvx) registry types are supported, and only
    when the package itself runs over stdio (a package whose own transport is
    http/sse expects a locally-allocated port — out of scope for v1). A
    package version of "latest" is not a pin at all, so falls back to the
    server's own version field, which is a real release number.
    """
    for pkg in packages:
        transport_type = (pkg.get("transport") or {}).get("type", "stdio")
        if transport_type != "stdio":
            continue
        registry_type = pkg.get("registryType")
        identifier = pkg.get("identifier")
        if not identifier:
            continue
        version = pkg.get("version") or ""
        if not version or version == "latest":
            version = server_version
        if not version:
            continue
        runtime_args = _literal_args(pkg.get("runtimeArguments"))
        package_args = _literal_args(pkg.get("packageArguments"))
        if runtime_args is None or package_args is None:
            logger.debug(
                "mcp_registry: skipping package %r — needs a runtime argument "
                "with no fixed value (unsupported in v1)",
                identifier,
            )
            continue
        inputs = _split_inputs(pkg.get("environmentVariables"))
        if inputs is None:
            logger.debug(
                "mcp_registry: skipping package %r — a required variable's default "
                "is an unresolvable template (unsupported in v1)",
                identifier,
            )
            continue
        required_inputs, literal_inputs = inputs
        if registry_type == "npm":
            args = ["-y", *runtime_args, f"{identifier}@{version}", *package_args]
            return "npx", args, required_inputs, literal_inputs
        if registry_type == "pypi":
            args = [*runtime_args, f"{identifier}=={version}", *package_args]
            return "uvx", args, required_inputs, literal_inputs
    return None


def _pick_remote(remotes: list[dict]) -> tuple[str, list[str], dict[str, str]] | None:
    """Return (url, required_header_names, literal_headers) for the first
    streamable-HTTP remote.

    SSE remotes are skipped explicitly rather than folded into "http" — v1
    only ever connects over streamable HTTP (see ``mcp_manager._connect``),
    and silently mislabeling an SSE-only server as "http" would produce a
    config that looks installed but can never actually connect.
    """
    for remote in remotes:
        transport_type = remote.get("type") or remote.get("transport_type")
        url = remote.get("url")
        if not url:
            continue
        if transport_type in ("streamable-http", "streamable_http", "http"):
            inputs = _split_inputs(remote.get("headers"))
            if inputs is None:
                logger.debug(
                    "mcp_registry: skipping remote %r — a required header's default "
                    "is an unresolvable template (unsupported in v1)",
                    url,
                )
                continue
            required_inputs, literal_inputs = inputs
            return url, required_inputs, literal_inputs
        if transport_type == "sse":
            logger.debug("mcp_registry: skipping SSE remote %r — unsupported transport", url)
    return None


def _split_inputs(entries: list[dict] | None) -> tuple[list[str], dict[str, str]] | None:
    """Split declared env vars / headers into ones needing a hushed secret at
    install time vs. ones with a literal, usable value.

    An entry carrying its own concrete ``value``/``default`` doesn't need a
    secret — even when marked required — since prompting for one would ask
    for a hushed binding on an input that already has a working value. That
    literal must still reach the installed server rather than being dropped
    on the floor, unless it's an unresolvable template like ``"{other_var}"``
    (a reference to some other input this v1 install flow has no way to
    substitute) — a *required* entry stuck in that state means the whole
    package/remote can't be installed as configured, so this returns None to
    signal "skip it entirely", matching ``_literal_args``' handling of an
    unsuppliable required argument.
    """
    if not entries:
        return [], {}
    required: list[str] = []
    literal: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        value = entry.get("value")
        if value in (None, ""):
            value = entry.get("default")
        if value not in (None, ""):
            value = str(value)
            if _TEMPLATE_RE.search(value):
                if entry.get("isRequired"):
                    return None
                continue
            literal[name] = value
            continue
        if entry.get("isRequired") or entry.get("isSecret"):
            required.append(name)
    return required, literal


def _literal_args(entries: list[dict] | None) -> list[str] | None:
    """Flatten a package's runtimeArguments/packageArguments into argv.

    Only entries with a fixed ``value`` or ``default`` can be represented —
    there's no v1 mechanism to collect an arbitrary user-supplied argument at
    install time. Returns None (skip the whole package) if a *required*
    argument has neither, since launching without it is known to be broken
    rather than merely incomplete.
    """
    if not entries:
        return []
    argv: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if value in (None, ""):
            value = entry.get("default")
        if value in (None, ""):
            if entry.get("isRequired"):
                return None
            continue
        if entry.get("type") == "named" and entry.get("name"):
            argv.extend([str(entry["name"]), str(value)])
        else:
            argv.append(str(value))
    return argv
