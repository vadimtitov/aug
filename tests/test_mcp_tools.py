"""Tests for aug/core/tools/mcp.py — search/install/list/remove MCP server tools."""

import subprocess
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aug.core.tools.mcp as mcp_tools
from aug.core.mcp_manager import McpServerHealth
from aug.core.tools.approval import ApprovalDecision
from aug.utils.file_settings import ApprovalRule, AppSettings, McpServerConfig, ToolSettings
from aug.utils.mcp_registry import McpRegistryServer
from aug.utils.state import AppState, McpCredentialBinding, McpInstallPlan

_P_APPROVAL = "aug.core.tools.approval.load_settings"
_APPROVE_ALL = AppSettings(tools=ToolSettings(approvals=[ApprovalRule(pattern=".*")]))


@contextmanager
def _patch_settings(settings: AppSettings, save_calls: list | None = None):
    """install_mcp_server's early "already configured" check calls the
    ``load_settings`` bound into aug.core.tools.mcp's own namespace, while
    update_settings() (aug/utils/file_settings.py) calls its *own* module's
    ``load_settings``/``save_settings`` internally — two independent
    bindings to the same underlying functions after `from ... import`, so
    both need patching for a consistent view across both call sites.
    """
    save_target = (
        patch("aug.utils.file_settings.save_settings", side_effect=lambda s: save_calls.append(s))
        if save_calls is not None
        else patch("aug.utils.file_settings.save_settings")
    )
    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.utils.file_settings.load_settings", return_value=settings),
        save_target,
    ):
        yield


def _server(
    name="io.github.x/server-postgres",
    transport="stdio",
    required_inputs=None,
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
        required_inputs=required_inputs or [],
    )


@pytest.fixture(autouse=True)
def _reset_state():
    mcp_tools._state.last_search = {}
    yield
    mcp_tools._state.last_search = {}


def _patch_no_plans():
    """Most tests don't care about install-plan persistence — this keeps
    _get_or_build_plan (via aug.utils.state.update_state) and _clear_plan
    (aug.core.tools.mcp's own load_state/save_state binding) both operating
    on the same in-memory state instead of touching disk — two independent
    bindings to the same underlying functions after `from ... import` (see
    _patch_settings above), so both modules need patching."""
    st = AppState()
    return (
        patch.multiple(
            "aug.core.tools.mcp",
            load_state=MagicMock(return_value=st),
            save_state=MagicMock(),
        ),
        patch.multiple(
            "aug.utils.state",
            load_state=MagicMock(return_value=st),
            save_state=MagicMock(),
        ),
    )


# ---------------------------------------------------------------------------
# search_mcp_servers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_mcp_servers_caches_snapshot_for_this_conversation():
    results = [_server()]
    with patch("aug.core.tools.mcp.McpRegistryClient.search", AsyncMock(return_value=results)):
        output = await mcp_tools.search_mcp_servers.ainvoke({"query": "postgres"})

    assert "server-postgres" in output
    assert mcp_tools._state.last_search[""] == results


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


@pytest.mark.asyncio
async def test_list_mcp_servers_none_configured():
    with patch("aug.core.tools.mcp.load_settings", return_value=AppSettings(mcp_servers=[])):
        output = await mcp_tools.list_mcp_servers.ainvoke({})
    assert "no mcp servers configured" in output.lower()


@pytest.mark.asyncio
async def test_list_mcp_servers_shows_active_health():
    cfg = McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])
    settings = AppSettings(mcp_servers=[cfg])
    manager = MagicMock()
    manager.health = {"postgres": McpServerHealth("postgres", "stdio", "active", tool_count=3)}

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=manager),
    ):
        output = await mcp_tools.list_mcp_servers.ainvoke({})

    assert "postgres" in output
    assert "active, 3 tools" in output


@pytest.mark.asyncio
async def test_list_mcp_servers_shows_failed_health():
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
        output = await mcp_tools.list_mcp_servers.ainvoke({})

    assert "failed (connection refused)" in output


@pytest.mark.asyncio
async def test_list_mcp_servers_disabled_server():
    cfg = McpServerConfig(name="old", transport="stdio", command="npx", args=[], enabled=False)
    settings = AppSettings(mcp_servers=[cfg])

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=None),
    ):
        output = await mcp_tools.list_mcp_servers.ainvoke({})

    assert "disabled" in output


@pytest.mark.asyncio
async def test_list_mcp_servers_pending_restart_when_no_manager_yet():
    cfg = McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])
    settings = AppSettings(mcp_servers=[cfg])

    with (
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.core.tools.mcp.get_manager", return_value=None),
    ):
        output = await mcp_tools.list_mcp_servers.ainvoke({})

    assert "pending restart" in output


# ---------------------------------------------------------------------------
# install_mcp_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_mcp_server_no_prior_search():
    with patch(_P_APPROVAL, return_value=_APPROVE_ALL), _patch_no_plans()[0], _patch_no_plans()[1]:
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})
    assert "run search_mcp_servers first" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_invalid_index():
    mcp_tools._state.last_search[""] = [_server()]
    load_p, save_p = _patch_no_plans()
    with patch(_P_APPROVAL, return_value=_APPROVE_ALL), load_p, save_p:
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 5})
    assert "no result #5" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_already_configured():
    mcp_tools._state.last_search[""] = [_server(name="io.github.x/server-postgres")]
    settings = AppSettings(
        mcp_servers=[
            McpServerConfig(name="postgres-0650d2", transport="stdio", command="npx", args=[])
        ]
    )
    load_p, save_p = _patch_no_plans()
    with patch(_P_APPROVAL, return_value=_APPROVE_ALL), _patch_settings(settings), load_p, save_p:
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})
    assert "already configured" in output.lower()


@pytest.mark.asyncio
async def test_install_mcp_server_missing_secrets_returns_plan_without_saving():
    mcp_tools._state.last_search[""] = [
        _server(name="io.github.x/server-github", required_inputs=["GITHUB_TOKEN"])
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []
    load_p, save_p = _patch_no_plans()

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        _patch_settings(settings, save_calls),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        load_p,
        save_p,
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "install plan" in output.lower()
    assert "GITHUB_TOKEN" in output
    assert "not set yet" in output.lower()
    assert save_calls == []


@pytest.mark.asyncio
async def test_install_mcp_server_shows_existing_secret_binding_for_approval():
    """Item 4: a registry-requested name that happens to match an existing
    hushed secret must be shown explicitly, not silently bound."""
    mcp_tools._state.last_search[""] = [
        _server(name="io.github.x/server-github", required_inputs=["GITHUB_TOKEN"])
    ]
    load_p, save_p = _patch_no_plans()
    with (
        load_p,
        save_p,
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"GITHUB_TOKEN"}),
    ):
        resource, operation = await mcp_tools._describe_install(1)

    assert resource == "github-0650d2"
    assert "GITHUB_TOKEN" in operation
    assert "existing hushed secret" in operation


@pytest.mark.asyncio
async def test_install_mcp_server_succeeds_when_secrets_present():
    mcp_tools._state.last_search[""] = [
        _server(name="io.github.x/server-github", required_inputs=["GITHUB_TOKEN"])
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []
    schedule_calls = []
    load_p, save_p = _patch_no_plans()

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        _patch_settings(settings, save_calls),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"GITHUB_TOKEN"}),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")) as mock_record,
        patch(
            "aug.core.tools.mcp._schedule_restart",
            side_effect=lambda op_id, name, interface, thread_id: schedule_calls.append(
                (op_id, name, interface, thread_id)
            ),
        ),
        load_p,
        save_p,
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "config saved" in output.lower()
    assert len(save_calls) == 1
    saved_cfg = save_calls[0].mcp_servers[0]
    assert saved_cfg.env == {"GITHUB_TOKEN": "hushed:GITHUB_TOKEN"}
    mock_record.assert_called_once_with(
        "install", "github-0650d2", "saved", interface="", thread_id=""
    )
    assert schedule_calls == [("op1", "github-0650d2", "", "")]


@pytest.mark.asyncio
async def test_install_mcp_server_sanitizes_header_name_into_secret_identifier():
    """Item 11: an HTTP header like "X-API-Key" is not a valid hushed secret
    name — installing must bind it to a sanitized identifier instead."""
    mcp_tools._state.last_search[""] = [
        _server(
            name="io.github.x/sentry-mcp",
            transport="http",
            required_inputs=["X-API-Key"],
        )
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []
    load_p, save_p = _patch_no_plans()

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        _patch_settings(settings, save_calls),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"X_API_KEY"}),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp._schedule_restart"),
        load_p,
        save_p,
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "config saved" in output.lower()
    saved_cfg = save_calls[0].mcp_servers[0]
    assert saved_cfg.headers == {"X-API-Key": "hushed:X_API_KEY"}


@pytest.mark.asyncio
async def test_install_mcp_server_no_required_secrets_installs_directly():
    mcp_tools._state.last_search[""] = [
        _server(name="io.github.x/server-simple", required_inputs=[])
    ]
    settings = AppSettings(mcp_servers=[])
    save_calls = []
    load_p, save_p = _patch_no_plans()

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        _patch_settings(settings, save_calls),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp._schedule_restart"),
        load_p,
        save_p,
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "config saved" in output.lower()
    assert len(save_calls) == 1


@pytest.mark.asyncio
async def test_install_mcp_server_denial_clears_the_plan():
    """Item 2 (denied plan lingers): denying an install must not leave its
    plan behind — otherwise a later search resolving the same index reuses
    the stale denied plan instead of the new search's result."""
    mcp_tools._state.last_search[""] = [_server(name="io.github.x/server-postgres")]
    state = AppState()

    # _get_or_build_plan (via aug.utils.state.update_state) and _clear_plan
    # (aug.core.tools.mcp's own load_state/save_state binding) must see the
    # same in-memory state — see _patch_settings's docstring for why both
    # modules' bindings need patching.
    with (
        patch("aug.core.tools.mcp.load_state", return_value=state),
        patch("aug.core.tools.mcp.save_state"),
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch(_P_APPROVAL, return_value=AppSettings()),  # no saved rule -> triggers interrupt
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.DENIED),
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "denied" in output.lower()
    assert state.mcp.install_plans == {}


@pytest.mark.asyncio
async def test_install_mcp_server_rechecks_duplicates_inside_lock():
    """Item 8: even if the fast-path check (against a snapshot loaded before
    the lock) passes, a duplicate that another writer already saved by the
    time the lock is actually held must still be caught before saving."""
    mcp_tools._state.last_search[""] = [_server(name="io.github.x/server-simple")]
    load_p, save_p = _patch_no_plans()

    # By the time update_settings() reloads under the lock, another writer
    # has already installed the same server.
    installed = AppSettings(
        mcp_servers=[McpServerConfig(name="simple-0650d2", transport="stdio", command="npx")]
    )

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", return_value=AppSettings(mcp_servers=[])),
        patch("aug.utils.file_settings.load_settings", return_value=installed),
        patch("aug.utils.file_settings.save_settings") as mock_save,
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp.update_operation_state", AsyncMock()),
        load_p,
        save_p,
    ):
        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1})

    assert "already configured" in output.lower()
    # update_settings() always persists on exit (even a no-op rewrite), but
    # the point of the recheck is that no *duplicate* entry gets appended.
    if mock_save.call_args is not None:
        assert len(mock_save.call_args[0][0].mcp_servers) == 1


# ---------------------------------------------------------------------------
# install plan persistence — approval interrupt/resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_build_plan_reuses_persisted_plan_across_a_different_search():
    """Regression test for the P1: an in-progress install must not be resolved
    against a different, later search snapshot — even the same process's own
    later search for a *different* thread must not interfere, and a persisted
    plan takes priority over whatever the live snapshot currently holds."""
    server = _server(name="io.github.x/server-postgres")
    mcp_tools._state.last_search["thread-a"] = [server]

    state = AppState()
    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
    ):
        plan = await mcp_tools._get_or_build_plan("thread-a", 1)
        assert plan is not None
        assert plan.slug == "postgres-0650d2"

        # Another conversation now searches and would, if scoping were
        # broken, shadow thread-a's in-flight install.
        mcp_tools._state.last_search["thread-b"] = [_server(name="io.github.x/server-mysql")]

        # thread-a resolving #1 again (LangGraph replaying the node on
        # resume) must get back the exact same persisted plan.
        again = await mcp_tools._get_or_build_plan("thread-a", 1)
    assert again == plan


@pytest.mark.asyncio
async def test_get_or_build_plan_survives_missing_live_snapshot():
    """Simulates a process restart while approval was pending: the in-memory
    search cache is gone, but the persisted plan (state.json) is not."""
    plan = McpInstallPlan(
        id="p1",
        thread_id="thread-a",
        search_index=1,
        server_name="io.github.x/server-postgres",
        slug="postgres",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "server-postgres@1.0.0"],
        credentials=[McpCredentialBinding(target_name="X", secret_name="X", bound=True)],
    )
    state = AppState()
    state.mcp.install_plans["thread-a"] = plan

    # No live snapshot at all (mcp_tools._state.last_search left empty by the fixture).
    # Secret "X" still present, so revalidation is a no-op on the plan itself
    # (update_state() still persists unconditionally on every call, same as
    # every other update_state() user in the codebase).
    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"X"}),
    ):
        resolved = await mcp_tools._get_or_build_plan("thread-a", 1)

    assert resolved == plan


@pytest.mark.asyncio
async def test_get_or_build_plan_revalidates_bound_status_on_retry():
    """Item 2 (missing-secret retry): a plan built while a secret was missing
    must notice once the user adds it and retries the same index — the old
    behavior reused the stale persisted plan forever, reporting the secret as
    still missing even after `hushed add` made it available."""
    plan = McpInstallPlan(
        id="p1",
        thread_id="thread-a",
        search_index=1,
        server_name="io.github.x/server-github",
        slug="github",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "server-github@1.0.0"],
        credentials=[
            McpCredentialBinding(
                target_name="GITHUB_TOKEN", secret_name="GITHUB_TOKEN", bound=False
            )
        ],
    )
    state = AppState()
    state.mcp.install_plans["thread-a"] = plan
    saved = []

    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state", side_effect=lambda s: saved.append(s)),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value={"GITHUB_TOKEN"}),
    ):
        resolved = await mcp_tools._get_or_build_plan("thread-a", 1)

    assert resolved.credentials[0].bound is True
    assert len(saved) == 1
    assert saved[0].mcp.install_plans["thread-a"].credentials[0].bound is True


@pytest.mark.asyncio
async def test_build_plan_preserves_literal_inputs():
    """Item 7: registry-declared literal/default values must survive into the
    persisted plan (and from there into the saved config's env_static)."""
    server = _server(name="io.github.x/server-simple")
    server = server.model_copy(update={"literal_inputs": {"LOG_LEVEL": "info"}})

    plan = mcp_tools._build_plan("thread-a", 1, server, known_secrets=set())

    assert plan.literal_inputs == {"LOG_LEVEL": "info"}
    cfg = mcp_tools._plan_to_config(plan, bindings={})
    assert cfg.env_static == {"LOG_LEVEL": "info"}


# ---------------------------------------------------------------------------
# remove_mcp_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_mcp_server_not_found():
    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        _patch_settings(AppSettings(mcp_servers=[])),
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
        _patch_settings(settings, save_calls),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op2")) as mock_record,
        patch(
            "aug.core.tools.mcp._schedule_restart",
            side_effect=lambda op_id, name, interface, thread_id: schedule_calls.append(
                (op_id, name, interface, thread_id)
            ),
        ),
    ):
        output = await mcp_tools.remove_mcp_server.ainvoke({"name": "postgres"})

    assert "removed" in output.lower()
    assert save_calls[0].mcp_servers == []
    mock_record.assert_called_once_with("remove", "postgres", "saved", interface="", thread_id="")
    assert schedule_calls == [("op2", "postgres", "", "")]


# ---------------------------------------------------------------------------
# _list_hushed_secrets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_hushed_secrets_parses_names():
    result = MagicMock(returncode=0, stdout="GITHUB_TOKEN\nDATABASE_URL\n", stderr="")
    with patch("aug.core.tools.mcp.subprocess.run", return_value=result):
        names = await mcp_tools._list_hushed_secrets()
    assert names == {"GITHUB_TOKEN", "DATABASE_URL"}


@pytest.mark.asyncio
async def test_list_hushed_secrets_returns_empty_on_failure():
    with patch(
        "aug.core.tools.mcp.subprocess.run", side_effect=subprocess.TimeoutExpired("hushed", 10)
    ):
        assert await mcp_tools._list_hushed_secrets() == set()


@pytest.mark.asyncio
async def test_list_hushed_secrets_returns_empty_on_nonzero_exit():
    result = MagicMock(returncode=1, stdout="", stderr="hushed: not found")
    with patch("aug.core.tools.mcp.subprocess.run", return_value=result):
        assert await mcp_tools._list_hushed_secrets() == set()


# ---------------------------------------------------------------------------
# _secret_name_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_name,expected",
    [
        ("DATABASE_URL", "DATABASE_URL"),
        ("X-API-Key", "X_API_KEY"),
        ("Authorization", "AUTHORIZATION"),
    ],
)
def test_secret_name_for_sanitizes(target_name, expected):
    assert mcp_tools._secret_name_for(target_name) == expected


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
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "not configured" in result.lower()
    assert update_calls == [("op1", "restart_pending")]


@pytest.mark.asyncio
async def test_trigger_restart_success_marks_pending_before_the_actual_restart_call():
    """Item 6: the durable state must flip to restart_pending *before* the
    restart is actually requested — that request is what's expected to kill
    this very process moments later."""
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(return_value={"Id": 1})
    mock_client.find_container_id = AsyncMock(return_value="abc123")

    order = []
    mock_client.container_action = AsyncMock(side_effect=lambda *a: order.append("restart"))

    with (
        patch("aug.core.tools.mcp.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.tools.mcp.update_operation_state",
            AsyncMock(side_effect=lambda *a: order.append(a)),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "restart triggered" in result.lower()
    assert order == [("op1", "restart_pending"), "restart"]


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
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
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
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
        ),
    ):
        result = await mcp_tools._trigger_restart("op1", "postgres")

    assert "restart failed" in result.lower()
    assert "portainer down" in result
    assert update_calls[0][1] == "failed"
