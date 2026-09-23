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
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from aug.utils.file_settings import McpServerConfig

logger = logging.getLogger(__name__)

_REGISTRY_BASE = "https://registry.modelcontextprotocol.io"
_TIMEOUT = 15.0
_DEFAULT_LIMIT = 10


class McpRegistryServer(BaseModel):
    """One search result, reduced to what installing it needs.

    ``required_env`` names environment variables (stdio) or header names
    (http) the server needs at runtime. Values are never part of a registry
    listing — they come from hushed, resolved at install/startup time.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    namespace: str
    description: str
    version: str
    transport: Literal["stdio", "http"]
    command: str = ""  # stdio only
    args: list[str] = []  # stdio only — includes the pinned version
    url: str = ""  # http only
    required_env: list[str] = []

    @property
    def slug(self) -> str:
        """Config + tool-namespace-safe short name, e.g. 'server-postgres' -> 'postgres'."""
        tail = self.name.rsplit("/", 1)[-1]
        return tail.removeprefix("server-") if tail.startswith("server-") else tail

    def to_config(self, env_refs: dict[str, str]) -> McpServerConfig:
        """Build the settings.json entry for this server.

        ``env_refs`` maps each required var/header name to its ``hushed:KEY``
        reference — built by the caller once it has confirmed the secret exists.
        """
        if self.transport == "stdio":
            return McpServerConfig(
                name=self.slug,
                transport="stdio",
                command=self.command,
                args=self.args,
                env=env_refs,
                enabled=True,
            )
        return McpServerConfig(
            name=self.slug,
            transport="http",
            url=self.url,
            headers=env_refs,
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
        command, args, required_env = package
        return McpRegistryServer(
            name=name,
            namespace=namespace,
            description=description,
            version=server_version,
            transport="stdio",
            command=command,
            args=args,
            required_env=required_env,
        )

    remote = _pick_remote(raw.get("remotes") or [])
    if remote is not None:
        url, required_env = remote
        return McpRegistryServer(
            name=name,
            namespace=namespace,
            description=description,
            version=server_version,
            transport="http",
            url=url,
            required_env=required_env,
        )

    logger.debug("mcp_registry: skipping %r — no installable package or remote", name)
    return None


def _pick_package(
    packages: list[dict], server_version: str
) -> tuple[str, list[str], list[str]] | None:
    """Return (command, args, required_env) for the first supported package.

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
        required_env = _required_names(pkg.get("environmentVariables"))
        if registry_type == "npm":
            return "npx", ["-y", f"{identifier}@{version}"], required_env
        if registry_type == "pypi":
            return "uvx", [f"{identifier}=={version}"], required_env
    return None


def _pick_remote(remotes: list[dict]) -> tuple[str, list[str]] | None:
    """Return (url, required_header_names) for the first http/streamable-http remote."""
    for remote in remotes:
        transport_type = remote.get("type") or remote.get("transport_type")
        url = remote.get("url")
        if not url:
            continue
        if transport_type in ("streamable-http", "streamable_http", "http", "sse"):
            return url, _required_names(remote.get("headers"))
    return None


def _required_names(entries: list[dict] | None) -> list[str]:
    """Extract the names of required (or secret) variables/headers from a list of
    registry ``{"name": ..., "isRequired": ..., "isSecret": ...}``-shaped dicts."""
    if not entries:
        return []
    names = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("name")
        if entry_name and (entry.get("isRequired") or entry.get("isSecret")):
            names.append(entry_name)
    return names
