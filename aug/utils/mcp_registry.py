"""Typed client for the official MCP Registry API (registry.modelcontextprotocol.io).

Parsing is deliberately tolerant — a field the registry renames or drops should
degrade one listing, not crash the whole search. Only stdio (npm/pypi via
npx/uvx) and remote HTTP servers are represented; Docker-based packages are
out of scope for v1 (no Docker socket access).
"""

import hashlib
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
# Length of the namespace-hash suffix in `McpRegistryServer.slug` — see its
# docstring for why a hash beats folding the namespace path in as text.
_SLUG_HASH_LEN = 6


class McpRegistryServer(BaseModel):
    """One search result, reduced to what installing it needs. ``required_inputs``
    names the runtime inputs the server needs (env vars for stdio, HTTP headers
    for http) — never themselves hushed secret names, see ``to_config``.
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
        Two namespaces can publish identically-named packages, and folding the
        namespace into the slug as text can itself collide — a hash suffix of
        the full namespace disambiguates without that risk.
        """
        tail = self.name.rsplit("/", 1)[-1]
        tail = tail.removeprefix("server-") if tail.startswith("server-") else tail
        if not self.namespace:
            return tail
        suffix = hashlib.sha256(self.namespace.encode()).hexdigest()[:_SLUG_HASH_LEN]
        return f"{tail}-{suffix}"

    def to_config(self, bindings: dict[str, str]) -> McpServerConfig:
        """Build the settings.json entry for this server. ``bindings`` maps each
        ``required_inputs`` name to its ``hushed:KEY`` reference."""
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
    """Best-effort parse of one registry entry. Prefers a stdio package (npm ->
    npx, pypi -> uvx), falls back to the first remote HTTP entry, else skips —
    one unsupported listing must never sink the whole search.
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
    supported package — npm/pypi over stdio only. A "latest" version isn't a
    real pin, so falls back to the server's own version field.
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
    streamable-HTTP remote. SSE remotes are skipped explicitly rather than
    folded into "http" — v1 only ever connects over streamable HTTP.
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
    """Split declared env vars / headers into ones needing a hushed secret vs.
    ones with a literal, usable value. A *required* entry stuck with an
    unresolvable template default (e.g. "{other_var}") means the whole
    package/remote can't be installed, so this returns None to skip it.
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
    """Flatten a package's runtimeArguments/packageArguments into argv. Returns
    None (skip the whole package) if a *required* argument has no fixed value.
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
