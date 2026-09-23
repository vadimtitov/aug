"""Tests for aug/utils/mcp_registry.py — MCP Registry API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aug.utils.file_settings import McpServerConfig
from aug.utils.mcp_registry import McpRegistryClient, McpRegistryServer, _parse_server

# ---------------------------------------------------------------------------
# _parse_server
# ---------------------------------------------------------------------------


def test_parse_server_npm_package():
    raw = {
        "name": "io.github.modelcontextprotocol/server-postgres",
        "description": "PostgreSQL MCP server",
        "version": "1.2.0",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "@modelcontextprotocol/server-postgres",
                "version": "1.2.0",
                "transport": {"type": "stdio"},
                "environmentVariables": [
                    {"name": "DATABASE_URL", "isRequired": True, "isSecret": True}
                ],
            }
        ],
    }
    server = _parse_server(raw)
    assert server is not None
    assert server.transport == "stdio"
    assert server.command == "npx"
    assert server.args == ["-y", "@modelcontextprotocol/server-postgres@1.2.0"]
    assert server.required_env == ["DATABASE_URL"]
    assert server.namespace == "io.github.modelcontextprotocol"
    assert server.slug == "postgres"


def test_parse_server_pypi_package():
    raw = {
        "name": "io.github.example/some-tool",
        "description": "A tool",
        "version": "0.3.0",
        "packages": [
            {
                "registryType": "pypi",
                "identifier": "some-mcp-tool",
                "version": "0.3.0",
                "transport": {"type": "stdio"},
            }
        ],
    }
    server = _parse_server(raw)
    assert server is not None
    assert server.command == "uvx"
    assert server.args == ["some-mcp-tool==0.3.0"]
    assert server.required_env == []


def test_parse_server_package_without_transport_key_defaults_to_stdio():
    """Older schema versions predate the "transport" field entirely."""
    raw = {
        "name": "io.github.example/legacy",
        "description": "pre-transport-field package",
        "version": "1.0.0",
        "packages": [{"registryType": "npm", "identifier": "legacy-pkg", "version": "1.0.0"}],
    }
    server = _parse_server(raw)
    assert server is not None
    assert server.transport == "stdio"


def test_parse_server_skips_http_transport_packages():
    """A package that itself listens over HTTP needs port allocation — out of
    scope for v1; the parser should fall through rather than mis-treat it as stdio."""
    raw = {
        "name": "io.github.example/http-package",
        "description": "runs a local http server",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "http-pkg",
                "version": "1.0.0",
                "transport": {"type": "streamable-http", "url": "http://127.0.0.1:{port}/mcp"},
            }
        ],
    }
    assert _parse_server(raw) is None


def test_parse_server_package_version_latest_falls_back_to_server_version():
    """A package pinned to the "latest" tag is not really pinned — fall back to
    the server's own concrete version field."""
    raw = {
        "name": "io.github.example/floating",
        "description": "unpinned package version",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "floating-pkg",
                "version": "latest",
                "transport": {"type": "stdio"},
            }
        ],
    }
    server = _parse_server(raw)
    assert server.args == ["-y", "floating-pkg@1.0.0"]


def test_parse_server_http_remote():
    raw = {
        "name": "io.github.sentry/sentry-mcp",
        "description": "Sentry MCP",
        "version": "2.0.0",
        "remotes": [
            {
                "type": "streamable-http",
                "url": "https://mcp.sentry.dev/mcp",
                "headers": [{"name": "Authorization", "isRequired": True, "isSecret": True}],
            }
        ],
    }
    server = _parse_server(raw)
    assert server is not None
    assert server.transport == "http"
    assert server.url == "https://mcp.sentry.dev/mcp"
    assert server.required_env == ["Authorization"]


def test_parse_server_prefers_package_over_remote():
    raw = {
        "name": "io.github.example/both",
        "description": "has both",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "pkg",
                "version": "1.0.0",
                "transport": {"type": "stdio"},
            }
        ],
        "remotes": [{"type": "http", "url": "https://example.com/mcp"}],
    }
    server = _parse_server(raw)
    assert server.transport == "stdio"


def test_parse_server_skips_unsupported_registry_type():
    """Docker/NuGet packages (out of scope for v1) fall through."""
    raw = {
        "name": "io.github.example/docker-only",
        "description": "docker package",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "oci",
                "identifier": "some/image",
                "version": "1.0.0",
                "transport": {"type": "stdio"},
            }
        ],
    }
    assert _parse_server(raw) is None


def test_parse_server_missing_name_returns_none():
    assert _parse_server({"description": "no name"}) is None


def test_parse_server_no_installable_target_returns_none():
    raw = {"name": "io.github.example/nothing", "description": "nothing installable"}
    assert _parse_server(raw) is None


def test_parse_server_no_env_vars_required():
    raw = {
        "name": "io.github.example/simple",
        "description": "no secrets needed",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "simple-server",
                "version": "1.0.0",
                "transport": {"type": "stdio"},
            }
        ],
    }
    server = _parse_server(raw)
    assert server.required_env == []


def test_slug_strips_server_prefix():
    server = McpRegistryServer(
        name="io.github.x/server-github",
        namespace="io.github.x",
        description="",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "pkg@1.0.0"],
    )
    assert server.slug == "github"


def test_slug_keeps_name_without_server_prefix():
    server = McpRegistryServer(
        name="io.github.x/dbhub",
        namespace="io.github.x",
        description="",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "pkg@1.0.0"],
    )
    assert server.slug == "dbhub"


def test_to_config_stdio():
    server = McpRegistryServer(
        name="io.github.x/server-github",
        namespace="io.github.x",
        description="",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github@1.0.0"],
        required_env=["GITHUB_PERSONAL_ACCESS_TOKEN"],
    )
    cfg = server.to_config({"GITHUB_PERSONAL_ACCESS_TOKEN": "hushed:GITHUB_PERSONAL_ACCESS_TOKEN"})
    assert isinstance(cfg, McpServerConfig)
    assert cfg.name == "github"
    assert cfg.transport == "stdio"
    assert cfg.command == "npx"
    assert cfg.env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "hushed:GITHUB_PERSONAL_ACCESS_TOKEN"}


def test_to_config_http():
    server = McpRegistryServer(
        name="io.github.sentry/sentry-mcp",
        namespace="io.github.sentry",
        description="",
        version="1.0.0",
        transport="http",
        url="https://mcp.sentry.dev/mcp",
        required_env=["Authorization"],
    )
    cfg = server.to_config({"Authorization": "hushed:SENTRY_BEARER_TOKEN"})
    assert cfg.transport == "http"
    assert cfg.url == "https://mcp.sentry.dev/mcp"
    assert cfg.headers == {"Authorization": "hushed:SENTRY_BEARER_TOKEN"}


# ---------------------------------------------------------------------------
# McpRegistryClient.search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_parsed_servers():
    # The live registry wraps each result as {"server": {...}, "_meta": {...}}.
    payload = {
        "servers": [
            {
                "server": {
                    "name": "io.github.x/server-postgres",
                    "description": "pg",
                    "version": "1.0.0",
                    "packages": [
                        {
                            "registryType": "npm",
                            "identifier": "server-postgres",
                            "version": "1.0.0",
                            "transport": {"type": "stdio"},
                        }
                    ],
                },
                "_meta": {"io.modelcontextprotocol.registry/official": {"status": "active"}},
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("aug.utils.mcp_registry.httpx.AsyncClient", return_value=mock_client):
        results = await McpRegistryClient().search("postgres")

    assert len(results) == 1
    assert results[0].slug == "postgres"


@pytest.mark.asyncio
async def test_search_skips_unparseable_entries_without_failing():
    payload = {
        "servers": [
            {"server": {"description": "malformed, no name"}},
            {"server": {"name": "io.github.x/nothing"}},
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("aug.utils.mcp_registry.httpx.AsyncClient", return_value=mock_client):
        results = await McpRegistryClient().search("x")

    assert results == []


@pytest.mark.asyncio
async def test_search_empty_results():
    mock_response = MagicMock()
    mock_response.json.return_value = {"servers": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("aug.utils.mcp_registry.httpx.AsyncClient", return_value=mock_client):
        results = await McpRegistryClient().search("nonexistent-xyz")

    assert results == []


@pytest.mark.asyncio
async def test_search_propagates_transport_errors():
    """The tool layer is responsible for turning this into an honest error string."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aug.utils.mcp_registry.httpx.AsyncClient", return_value=mock_client),
        pytest.raises(httpx.ConnectError),
    ):
        await McpRegistryClient().search("x")


@pytest.mark.asyncio
async def test_search_raises_on_http_error_status():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aug.utils.mcp_registry.httpx.AsyncClient", return_value=mock_client),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await McpRegistryClient().search("x")
