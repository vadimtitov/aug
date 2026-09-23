"""Tests for aug/core/tools/mcp.py — search/install/list/remove MCP server tools."""

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aug.core.tools.mcp as mcp_tools
from aug.core.mcp_manager import McpServerHealth
from aug.utils.file_settings import ApprovalRule, AppSettings, McpServerConfig, ToolSettings
from aug.utils.mcp_registry import McpRegistryServer

_P_APPROVAL = "aug.core.tools.approval.load_settings"
_APPROVE_ALL = AppSettings(tools=ToolSettings(approvals=[ApprovalRule(pattern=".*")]))


def _server(
    name="io.github.x/server-postgres",
    transport="stdio",
    required_env=None,
) -> McpRegistryServer:
    return McpRegistryServer(
        name=name,
        namespace="io.github.x",
        description="A test server",
        version="1.0.0",
        transport=transport,
        command="npx",
        args=["-y", "server-postgres@1.0.0"],
        url="https://example.com/mcp" if transport == "http" else "",
        required_env=required_env or [],
    )


@pytest.fixture(autouse=True)
def _reset_search_cache():
    mcp_tools._last_search = []
    yield
    mcp_tools._last_search = []


# ---------------------------------------------------------------------------
# search_mcp_servers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_mcp_servers_caches_snapshot():
    results = [_server()]
    with patch("aug.core.tools.mcp.McpRegistryClient.search", AsyncMock(return_value=results)):
        output = await mcp_tools.search_mcp_servers.ainvoke({"query": "postgres"})

    assert "server-postgres" in output
    assert mcp_tools._last_search == results


@pytest.mark.asyncio
async def test_search_mcp_servers_no_results():
    with patch("aug.core.tools.mcp.McpRegistryClient.search", AsyncMock(return_value=[])):
        output = await mcp_tools.search_mcp_servers.ainvoke({"query": "nonexistent"})
    assert "no mcp servers found" in output.lower()


@pytest.mark.asyncio
async def test_search_mcp_servers_reports_failure_honestly():
    with patch(
        "aug.core.tools.mcp.McpRegistryClient.search",
        AsyncMock(side_effect=RuntimeError("registry unreachable")),
    ):
        output = await mcp_tools.search_mcp_servers.ainvoke({"query": "x"})
    assert "did not complete" in output.lower()
    assert "registry unreachable" in output


# ---------------------------------------------------------------------------
# list_mcp_servers
# ---------------------------------------------------------------------------


def test_list_mcp_servers_none_configured():
    with patch("aug.core.tools.mcp.load_settings", return_value=AppSettings(mcp_servers=[])):
        output = mcp_tools.list_mcp_servers.invoke({})
    assert "no mcp servers configured" in output.lower()


def test_list_mcp_servers_shows_active_health():
    cfg = McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])
    settings = AppSettings(mcp_servers=[cfg])
    manager = MagicMock()
    manager.health = {"postgres": McpServerHealth("postgres", "stdio", "active", tool_count=3)}

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=manager),
    ):
        output = mcp_tools.list_mcp_servers.invoke({})

    assert "postgres" in output
    assert "active, 3 tools" in output


def test_list_mcp_servers_shows_failed_health():
    cfg = McpServerConfig(name="sentry", transport="http", url="https://x")
    settings = AppSettings(mcp_servers=[cfg])
    manager = MagicMock()
    manager.health = {
        "sentry": McpServerHealth("sentry", "http", "failed", error="connection refused")
    }

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=manager),
    ):
        output = mcp_tools.list_mcp_servers.invoke({})

    assert "failed (connection refused)" in output


def test_list_mcp_servers_disabled_server():
    cfg = McpServerConfig(name="old", transport="stdio", command="npx", args=[], enabled=False)
    settings = AppSettings(mcp_servers=[cfg])

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=None),
    ):
        output = mcp_tools.list_mcp_servers.invoke({})

    assert "disabled" in output


def test_list_mcp_servers_pending_restart_when_no_manager_yet():
    cfg = McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])
    settings = AppSettings(mcp_servers=[cfg])

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=None),
    ):
        output = mcp_tools.list_mcp_servers.invoke({})

    assert "pending restart" in output


# ---------------------------------------------------------------------------
# install_mcp_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_mcp_server_no_prior_search():
    with patch(_P_APPROVAL, return_value=_APPROVE_ALL):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})
    assert "run search_mcp_servers first" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_invalid_index():
    mcp_tools._last_search = [_server()]
    with patch(_P_APPROVAL, return_value=_APPROVE_ALL):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 5})
    assert "no result #5" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_already_configured():
    mcp_tools._last_search = [_server(name="io.github.x/server-postgres")]
    settings = AppSettings(
        mcp_servers=[McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])]
    )
    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})
    assert "already configured" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_missing_secrets_returns_plan_without_saving():
    mcp_tools._last_search = [
        _server(name="io.github.x/server-github", required_env=["GITHUB_TOKEN"])
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.save_settings", side_effect=lambda s: save_calls.append(s)),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "install plan" in output.lower()
    assert "hushed add GITHUB_TOKEN" in output
    assert save_calls == []


@pytest.mark.asyncio
async def test_install_mcp_server_succeeds_when_secrets_present():
    mcp_tools._last_search = [
        _server(name="io.github.x/server-github", required_env=["GITHUB_TOKEN"])
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []
    schedule_calls = []

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.save_settings", side_effect=lambda s: save_calls.append(s)),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"GITHUB_TOKEN"}),
        patch("aug.core.tools.mcp.record_operation", return_value="op1") as mock_record,
        patch(
            "aug.core.tools.mcp._schedule_restart",
            side_effect=lambda op_id, name: schedule_calls.append((op_id, name)),
        ),
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "config saved" in output.lower()
    assert len(save_calls) == 1
    saved_cfg = save_calls[0].mcp_servers[0]
    assert saved_cfg.env == {"GITHUB_TOKEN": "hushed:GITHUB_TOKEN"}
    mock_record.assert_called_once_with("install", "github", "saved")
    assert schedule_calls == [("op1", "github")]


@pytest.mark.asyncio
async def test_install_mcp_server_no_required_secrets_installs_directly():
    mcp_tools._last_search = [_server(name="io.github.x/server-simple", required_env=[])]
    settings = AppSettings(mcp_servers=[])
    save_calls = []

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.save_settings", side_effect=lambda s: save_calls.append(s)),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch("aug.core.tools.mcp.record_operation", return_value="op1"),
        patch("aug.core.tools.mcp._schedule_restart"),
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "config saved" in output.lower()
    assert len(save_calls) == 1


# ---------------------------------------------------------------------------
# remove_mcp_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_mcp_server_not_found():
    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=AppSettings(mcp_servers=[])),
    ):
        output = await mcp_tools.remove_mcp_server.ainvoke({"name": "ghost"})
    assert "no mcp server named" in output.lower()


@pytest.mark.asyncio
async def test_remove_mcp_server_success():
    settings = AppSettings(
        mcp_servers=[McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])]
    )
    save_calls = []
    schedule_calls = []

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.save_settings", side_effect=lambda s: save_calls.append(s)),
        patch("aug.core.tools.mcp.record_operation", return_value="op2") as mock_record,
        patch(
            "aug.core.tools.mcp._schedule_restart",
            side_effect=lambda op_id, name: schedule_calls.append((op_id, name)),
        ),
    ):
        output = await mcp_tools.remove_mcp_server.ainvoke({"name": "postgres"})

    assert "removed" in output.lower()
    assert save_calls[0].mcp_servers == []
    mock_record.assert_called_once_with("remove", "postgres", "saved")
    assert schedule_calls == [("op2", "postgres")]


# ---------------------------------------------------------------------------
# _list_hushed_secrets
# ---------------------------------------------------------------------------


def test_list_hushed_secrets_parses_names():
    result = MagicMock(returncode=0, stdout="GITHUB_TOKEN\nDATABASE_URL\n", stderr="")
    with patch("aug.core.tools.mcp.subprocess.run", return_value=result):
        names = mcp_tools._list_hushed_secrets()
    assert names == {"GITHUB_TOKEN", "DATABASE_URL"}


def test_list_hushed_secrets_returns_empty_on_failure():
    with patch(
        "aug.core.tools.mcp.subprocess.run", side_effect=subprocess.TimeoutExpired("hushed", 10)
    ):
        assert mcp_tools._list_hushed_secrets() == set()


def test_list_hushed_secrets_returns_empty_on_nonzero_exit():
    result = MagicMock(returncode=1, stdout="", stderr="hushed: not found")
    with patch("aug.core.tools.mcp.subprocess.run", return_value=result):
        assert mcp_tools._list_hushed_secrets() == set()


# ---------------------------------------------------------------------------
# _trigger_restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_restart_portainer_not_configured():
    update_calls = []
    mock_client = MagicMock()
    mock_client.is_configured.return_value = False

    with (
        patch("aug.core.tools.mcp.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.tools.mcp.update_operation_state",
            side_effect=lambda *a: update_calls.append(a),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "not configured" in result.lower()
    assert update_calls == [("op1", "restart_pending")]


@pytest.mark.asyncio
async def test_trigger_restart_success():
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(return_value={"Id": 1})
    mock_client.find_container_id = AsyncMock(return_value="abc123")
    mock_client.container_action = AsyncMock()

    update_calls = []
    with (
        patch("aug.core.tools.mcp.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.tools.mcp.update_operation_state",
            side_effect=lambda *a: update_calls.append(a),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    mock_client.container_action.assert_awaited_once_with("abc123", 1, "restart")
    assert "restart triggered" in result.lower()
    assert update_calls == [("op1", "restart_pending")]


@pytest.mark.asyncio
async def test_trigger_restart_container_not_found():
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(return_value={"Id": 1})
    mock_client.find_container_id = AsyncMock(return_value=None)

    update_calls = []
    with (
        patch("aug.core.tools.mcp.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.tools.mcp.update_operation_state",
            side_effect=lambda *a: update_calls.append(a),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "restart failed" in result.lower()
    assert update_calls[0][1] == "failed"


@pytest.mark.asyncio
async def test_trigger_restart_handles_exception():
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(side_effect=RuntimeError("portainer down"))

    update_calls = []
    with (
        patch("aug.core.tools.mcp.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.tools.mcp.update_operation_state",
            side_effect=lambda *a: update_calls.append(a),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "restart failed" in result.lower()
    assert "portainer down" in result
    assert update_calls[0][1] == "failed"
