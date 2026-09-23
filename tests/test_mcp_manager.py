"""Tests for aug/core/mcp_manager.py — MCPManager, secret resolution, namespacing."""

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel

from aug.core import mcp_manager
from aug.core.mcp_manager import (
    MCPManager,
    McpOperationOutcome,
    McpSecretError,
    McpServerHealth,
    _namespace_tool,
    _read_hushed_secret,
    _resolve_hushed_ref,
    _resolve_stdio_env,
    get_manager,
    record_operation,
    set_manager,
    update_operation_state,
)
from aug.utils.file_settings import AppSettings, McpServerConfig
from aug.utils.state import AppState, McpOperation

# ---------------------------------------------------------------------------
# _read_hushed_secret / _resolve_hushed_ref
# ---------------------------------------------------------------------------


def _fake_run_writes(value: bytes):
    """subprocess.run side_effect that writes *value* to the fd hushed would
    have written to, mimicking the real python-child invocation."""

    def _run(cmd, pass_fds=(), **kwargs):
        for fd in pass_fds:
            os.write(fd, value)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def test_read_hushed_secret_success():
    with patch(
        "aug.core.mcp_manager.subprocess.run", side_effect=_fake_run_writes(b"super-secret")
    ):
        value = _read_hushed_secret("MY_TOKEN")
    assert value == "super-secret"


def test_read_hushed_secret_never_reads_from_stdout():
    """Regression guard: hushed redacts secret values from captured stdout/stderr,
    so the resolver must not rely on `result.stdout` for the value."""

    def _run(cmd, pass_fds=(), **kwargs):
        for fd in pass_fds:
            os.write(fd, b"the-real-value")
        return subprocess.CompletedProcess(cmd, 0, stdout="[REDACTED]", stderr="")

    with patch("aug.core.mcp_manager.subprocess.run", side_effect=_run):
        value = _read_hushed_secret("MY_TOKEN")
    assert value == "the-real-value"


def test_read_hushed_secret_missing_raises():
    with (
        patch("aug.core.mcp_manager.subprocess.run", side_effect=_fake_run_writes(b"")),
        pytest.raises(McpSecretError, match="not set"),
    ):
        _read_hushed_secret("MISSING_TOKEN")


def test_read_hushed_secret_command_failure_raises():
    def _run(cmd, pass_fds=(), **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="hushed: boom")

    with (
        patch("aug.core.mcp_manager.subprocess.run", side_effect=_run),
        pytest.raises(McpSecretError, match="boom"),
    ):
        _read_hushed_secret("MY_TOKEN")


def test_read_hushed_secret_rejects_shell_metacharacters_in_name():
    """A secret name is no longer shell-interpolated at all (it's an argv element
    to a Python child), but it's still validated up front so a malformed name
    fails clearly instead of silently not matching anything."""
    with (
        patch("aug.core.mcp_manager.subprocess.run") as mock_run,
        pytest.raises(McpSecretError, match="invalid secret name"),
    ):
        _read_hushed_secret('TOKEN"; rm -rf / #')
    mock_run.assert_not_called()


def test_read_hushed_secret_timeout_raises():
    with (
        patch(
            "aug.core.mcp_manager.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="hushed", timeout=15),
        ),
        pytest.raises(McpSecretError),
    ):
        _read_hushed_secret("MY_TOKEN")


def test_read_hushed_secret_passes_fd_as_argv_not_shell_redirection():
    """Regression guard for the fd>=10 bug: the descriptor must travel as a
    plain argv element (int(sys.argv[1])), never interpolated into shell
    redirection syntax like `>&{fd}`, which silently breaks past descriptor 9
    on Debian's dash."""
    captured = {}

    def _run(cmd, pass_fds=(), **kwargs):
        captured["cmd"] = cmd
        for fd in pass_fds:
            os.write(fd, b"value")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("aug.core.mcp_manager.subprocess.run", side_effect=_run):
        _read_hushed_secret("MY_TOKEN")

    cmd = captured["cmd"]
    assert not any(">&" in part for part in cmd)
    assert "sh" not in cmd


@pytest.mark.asyncio
async def test_resolve_hushed_ref_rejects_non_hushed_prefix():
    with pytest.raises(McpSecretError, match="hushed:KEY"):
        await _resolve_hushed_ref("plaintext-value")


@pytest.mark.asyncio
async def test_resolve_hushed_ref_never_echoes_rejected_value():
    """A misconfigured plaintext secret must never appear in the error text —
    that's the exact value the check exists to keep out of logs."""
    with pytest.raises(McpSecretError) as excinfo:
        await _resolve_hushed_ref("super-secret-plaintext-token")
    assert "super-secret-plaintext-token" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolve_hushed_ref_strips_prefix_and_resolves():
    with patch(
        "aug.core.mcp_manager._read_hushed_secret", return_value="resolved-value"
    ) as mock_read:
        value = await _resolve_hushed_ref("hushed:MY_TOKEN")
    assert value == "resolved-value"
    mock_read.assert_called_once_with("MY_TOKEN")


# ---------------------------------------------------------------------------
# _resolve_stdio_env — scrubbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_stdio_env_only_includes_default_and_declared_vars():
    with (
        patch(
            "aug.core.mcp_manager.get_default_environment",
            return_value={"PATH": "/usr/bin", "HOME": "/root"},
        ),
        patch("aug.core.mcp_manager._read_hushed_secret", return_value="tok-value"),
    ):
        env = await _resolve_stdio_env({"GITHUB_PERSONAL_ACCESS_TOKEN": "hushed:GH_TOKEN"})

    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "GITHUB_PERSONAL_ACCESS_TOKEN": "tok-value",
    }
    # Never AUG's own process env (API_KEY etc.) — only what get_default_environment
    # returned plus explicitly declared, resolved secrets.
    assert "API_KEY" not in env


@pytest.mark.asyncio
async def test_resolve_stdio_env_empty_declared_still_scrubbed():
    with patch("aug.core.mcp_manager.get_default_environment", return_value={"PATH": "/usr/bin"}):
        env = await _resolve_stdio_env({})
    assert env == {"PATH": "/usr/bin"}


# ---------------------------------------------------------------------------
# _namespace_tool
# ---------------------------------------------------------------------------


class _EchoArgs(BaseModel):
    text: str = ""


def _make_tool(name: str, coroutine) -> StructuredTool:
    return StructuredTool(
        name=name,
        description="a tool",
        args_schema=_EchoArgs,
        coroutine=coroutine,
        response_format="content",
    )


@pytest.mark.asyncio
async def test_namespace_tool_renames_and_preserves_behavior():
    async def _echo(text: str = "") -> str:
        return f"echo:{text}"

    original = _make_tool("search", _echo)
    wrapped = _namespace_tool(original, "github__search")

    assert wrapped.name == "github__search"
    result = await wrapped.coroutine(text="hi")
    assert result == "echo:hi"


@pytest.mark.asyncio
async def test_namespace_tool_converts_timeout_to_tool_exception():
    """A timeout must become an explicit tool failure (ToolException,
    handle_tool_error=True), not a bare TimeoutError that escapes the
    compiled graph and aborts the whole agent run."""

    async def _hangs(text: str = "") -> str:
        await asyncio.sleep(10)
        return "never"

    original = _make_tool("slow", _hangs)
    wrapped = _namespace_tool(original, "server__slow")
    assert wrapped.handle_tool_error is True

    with patch("aug.core.mcp_manager._TOOL_CALL_TIMEOUT", 0.05):
        with pytest.raises(ToolException, match="did NOT complete"):
            await wrapped.coroutine(text="x")


@pytest.mark.asyncio
async def test_namespace_tool_wraps_other_failures_as_tool_exception():
    async def _boom(text: str = "") -> str:
        raise ConnectionError("pipe closed")

    original = _make_tool("flaky", _boom)
    wrapped = _namespace_tool(original, "server__flaky")

    with pytest.raises(ToolException, match="pipe closed"):
        await wrapped.coroutine(text="x")


# ---------------------------------------------------------------------------
# MCPManager.load_all / aclose — owner-task session lifecycle
# ---------------------------------------------------------------------------


def _stdio_cfg(name: str, enabled: bool = True) -> McpServerConfig:
    return McpServerConfig(
        name=name, transport="stdio", command="npx", args=["-y", f"{name}@1.0.0"], enabled=enabled
    )


class _TaskBoundSession:
    """Mimics AnyIO's real requirement that a cancel scope is exited by the
    same task that entered it — raises exactly the RuntimeError the review
    reproduced if __aexit__ runs on a different task than __aenter__."""

    def __init__(self) -> None:
        self._enter_task: asyncio.Task | None = None
        self.initialize = AsyncMock()
        self.closed_cleanly = False

    async def __aenter__(self):
        self._enter_task = asyncio.current_task()
        return self

    async def __aexit__(self, *exc):
        if asyncio.current_task() is not self._enter_task:
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than it was entered in"
            )
        self.closed_cleanly = True
        return False


def _session_factory(session: _TaskBoundSession):
    @asynccontextmanager
    async def _open(cfg):
        async with session as s:
            yield s

    return _open


def _fake_tool(name: str) -> StructuredTool:
    async def _noop(**kwargs) -> str:
        return "ok"

    return StructuredTool(
        name=name,
        description="d",
        args_schema=_EchoArgs,
        coroutine=_noop,
        response_format="content",
    )


@pytest.mark.asyncio
async def test_load_all_no_servers_is_a_noop():
    with patch("aug.core.mcp_manager.load_settings", return_value=AppSettings(mcp_servers=[])):
        manager = MCPManager()
        await manager.load_all()
    assert manager.tools == []
    assert manager.health == {}


@pytest.mark.asyncio
async def test_load_all_skips_disabled_servers():
    settings = AppSettings(mcp_servers=[_stdio_cfg("a", enabled=False)])
    with patch("aug.core.mcp_manager.load_settings", return_value=settings):
        manager = MCPManager()
        await manager.load_all()
    assert manager.health == {}


@pytest.mark.asyncio
async def test_connect_and_shutdown_same_task_owns_whole_lifecycle():
    """Regression test for the P1: entering and exiting a session's context in
    different tasks raises under real AnyIO. This uses a session double that
    enforces the exact same constraint, so load_all()+aclose() succeeding
    proves the owner task, not some other task, does both."""
    settings = AppSettings(mcp_servers=[_stdio_cfg("github")])
    session = _TaskBoundSession()

    with (
        patch("aug.core.mcp_manager.load_settings", return_value=settings),
        patch("aug.core.mcp_manager._open_session", _session_factory(session)),
        patch.object(MCPManager, "_namespaced_tools", AsyncMock(return_value=[_fake_tool("x")])),
    ):
        manager = MCPManager()
        await manager.load_all()
        assert manager.health["github"].status == "active"
        await manager.aclose()

    assert session.closed_cleanly is True


@pytest.mark.asyncio
async def test_load_all_isolates_per_server_failures():
    settings = AppSettings(mcp_servers=[_stdio_cfg("good"), _stdio_cfg("bad")])
    good_session = _TaskBoundSession()

    @asynccontextmanager
    async def _open(cfg):
        if cfg.name == "bad":
            raise RuntimeError("connection refused")
        async with good_session as s:
            yield s

    with (
        patch("aug.core.mcp_manager.load_settings", return_value=settings),
        patch("aug.core.mcp_manager._open_session", _open),
        patch.object(MCPManager, "_namespaced_tools", AsyncMock(return_value=[_fake_tool("x")])),
    ):
        manager = MCPManager()
        await manager.load_all()

    assert manager.health["good"].status == "active"
    assert manager.health["bad"].status == "failed"
    assert "connection refused" in manager.health["bad"].error
    assert len(manager.tools) == 1


@pytest.mark.asyncio
async def test_load_all_respects_overall_deadline():
    settings = AppSettings(mcp_servers=[_stdio_cfg("slow")])

    @asynccontextmanager
    async def _open(cfg):
        await asyncio.sleep(10)
        yield _TaskBoundSession()

    with (
        patch("aug.core.mcp_manager.load_settings", return_value=settings),
        patch("aug.core.mcp_manager._open_session", _open),
        patch("aug.core.mcp_manager._OVERALL_TIMEOUT", 0.05),
        patch("aug.core.mcp_manager._PER_SERVER_TIMEOUT", 5),
    ):
        manager = MCPManager()
        await manager.load_all()  # must not raise

    assert manager.tools == []


@pytest.mark.asyncio
async def test_load_one_timeout_abandons_and_cleans_up_in_owner_task():
    """A server that never calls initialize() in time must have its
    partially-initialized session closed immediately, by its own owner task
    (via cancellation) — not left dangling."""
    cfg = _stdio_cfg("slow")
    session = _TaskBoundSession()

    @asynccontextmanager
    async def _open(cfg):
        async with session as s:
            await asyncio.sleep(10)  # never reaches session.initialize()
            yield s

    manager = MCPManager()
    with (
        patch("aug.core.mcp_manager._open_session", _open),
        patch("aug.core.mcp_manager._PER_SERVER_TIMEOUT", 0.05),
    ):
        await manager._load_one(cfg)

    assert manager.health["slow"].status == "failed"
    assert session.closed_cleanly is True


@pytest.mark.asyncio
async def test_connect_namespaces_tools_and_detects_collision():
    cfg_a = _stdio_cfg("github")
    session = _TaskBoundSession()

    manager = MCPManager()
    manager._tool_names = {"github__search"}  # simulate a pre-existing collision

    async def _fake_namespaced_tools(cfg, sess):
        # Mirrors the real skip-on-collision behavior for this one input.
        return []

    with (
        patch("aug.core.mcp_manager._open_session", _session_factory(session)),
        patch.object(manager, "_namespaced_tools", side_effect=_fake_namespaced_tools),
    ):
        await manager._load_one(cfg_a)

    assert manager.health["github"].tool_count == 0
    assert manager.tools == []


@pytest.mark.asyncio
async def test_connect_success_namespaces_and_registers_tool():
    cfg = _stdio_cfg("github")
    session = _TaskBoundSession()

    manager = MCPManager()
    with (
        patch("aug.core.mcp_manager._open_session", _session_factory(session)),
        patch.object(
            manager, "_namespaced_tools", AsyncMock(return_value=[_fake_tool("github__search")])
        ),
    ):
        await manager._load_one(cfg)

    assert manager.health["github"].status == "active"
    assert manager.health["github"].tool_count == 1
    assert manager.tools[0].name == "github__search"


@pytest.mark.asyncio
async def test_namespaced_tools_skips_overlong_tool_name():
    manager = MCPManager()
    manager._tool_names = set()
    session = AsyncMock()

    with patch(
        "aug.core.mcp_manager.load_mcp_tools",
        AsyncMock(return_value=[_fake_tool("x" * 80)]),
    ):
        tools = await manager._namespaced_tools(_stdio_cfg("server"), session)

    assert tools == []


@pytest.mark.asyncio
async def test_load_one_wraps_secret_error_as_failed_health():
    cfg = _stdio_cfg("needs-secret")

    @asynccontextmanager
    async def _open(cfg):
        raise McpSecretError("secret 'X' is not set")
        yield  # pragma: no cover

    manager = MCPManager()
    with patch("aug.core.mcp_manager._open_session", _open):
        await manager._load_one(cfg)
    assert manager.health["needs-secret"].status == "failed"
    assert "not set" in manager.health["needs-secret"].error


# ---------------------------------------------------------------------------
# reconcile_operations
# ---------------------------------------------------------------------------


def _state_with_pending(action: str = "install", state: str = "restart_pending") -> AppState:
    st = AppState()
    st.mcp.operations.append(
        McpOperation(id="op1", action=action, server_name="postgres", state=state)
    )
    return st


@contextmanager
def _patch_state(state: AppState, saved: dict | None = None):
    """reconcile_operations() peeks via the module-bound load_state (a direct
    `from ... import` into mcp_manager.py) before taking the lock, then does
    its real read-modify-write through update_state() — which calls
    aug.utils.state's *own* load_state/save_state internally. Both bindings
    need to agree for a test to see a consistent view."""
    save_target = (
        patch("aug.utils.state.save_state", side_effect=lambda s: saved.update(state=s))
        if saved is not None
        else patch("aug.utils.state.save_state")
    )
    with (
        patch("aug.core.mcp_manager.load_state", return_value=state),
        patch("aug.utils.state.load_state", return_value=state),
        save_target,
    ):
        yield


@pytest.mark.asyncio
async def test_reconcile_operations_marks_install_active_when_healthy():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth("postgres", "stdio", "active", tool_count=3)

    saved = {}
    with _patch_state(_state_with_pending(), saved):
        outcomes = await manager.reconcile_operations()

    assert len(outcomes) == 1
    assert "succeeded" in outcomes[0].summary
    assert saved["state"].mcp.operations[0].state == "active"


@pytest.mark.asyncio
async def test_reconcile_operations_marks_install_failed_when_unhealthy():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth(
        "postgres", "stdio", "failed", error="connection refused"
    )

    saved = {}
    with _patch_state(_state_with_pending(), saved):
        outcomes = await manager.reconcile_operations()

    assert "failed" in outcomes[0].summary
    assert "connection refused" in outcomes[0].summary
    assert saved["state"].mcp.operations[0].state == "failed"


@pytest.mark.asyncio
async def test_reconcile_operations_recovers_operations_still_at_saved():
    """If AUG died before _trigger_restart ever wrote restart_pending, the
    operation is stuck at "saved" — reconcile must still pick it up, since the
    config was already written to disk and attempted this boot."""
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth("postgres", "stdio", "active", tool_count=1)

    with _patch_state(_state_with_pending(state="saved")):
        outcomes = await manager.reconcile_operations()

    assert len(outcomes) == 1
    assert "succeeded" in outcomes[0].summary


@pytest.mark.asyncio
async def test_reconcile_operations_removal_success_is_absence_not_failure():
    """A removed server correctly disappearing from health must be reported as
    success, not "server not found after restart"."""
    manager = MCPManager()  # no health entry at all — the server is gone, as intended

    with _patch_state(_state_with_pending(action="remove")):
        outcomes = await manager.reconcile_operations()

    assert len(outcomes) == 1
    assert "succeeded" in outcomes[0].summary
    assert "failed" not in outcomes[0].summary


@pytest.mark.asyncio
async def test_reconcile_operations_removal_failure_when_still_connected():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth("postgres", "stdio", "active", tool_count=1)

    with _patch_state(_state_with_pending(action="remove")):
        outcomes = await manager.reconcile_operations()

    assert "failed" in outcomes[0].summary


@pytest.mark.asyncio
async def test_reconcile_operations_returns_empty_when_nothing_pending():
    manager = MCPManager()
    with _patch_state(AppState()):
        assert await manager.reconcile_operations() == []


@pytest.mark.asyncio
async def test_reconcile_operations_carries_interface_and_thread_id():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth("postgres", "stdio", "active", tool_count=1)
    st = AppState()
    st.mcp.operations.append(
        McpOperation(
            id="op1",
            action="install",
            server_name="postgres",
            state="restart_pending",
            interface="telegram",
            thread_id="tg-123",
        )
    )

    with _patch_state(st):
        outcomes = await manager.reconcile_operations()

    assert isinstance(outcomes[0], McpOperationOutcome)
    assert outcomes[0].interface == "telegram"
    assert outcomes[0].thread_id == "tg-123"


# ---------------------------------------------------------------------------
# record_operation / update_operation_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_operation_persists_and_returns_id():
    saved = {}
    with (
        patch("aug.utils.state.load_state", return_value=AppState()),
        patch("aug.utils.state.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        op_id = await record_operation(
            "install", "postgres", "saved", interface="telegram", thread_id="tg-1"
        )

    assert op_id
    op = saved["state"].mcp.operations[0]
    assert op.id == op_id
    assert op.action == "install"
    assert op.server_name == "postgres"
    assert op.state == "saved"
    assert op.interface == "telegram"
    assert op.thread_id == "tg-1"


@pytest.mark.asyncio
async def test_update_operation_state_updates_matching_record():
    st = AppState()
    st.mcp.operations.append(
        McpOperation(id="op1", action="install", server_name="postgres", state="saved")
    )
    saved = {}
    with (
        patch("aug.utils.state.load_state", return_value=st),
        patch("aug.utils.state.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        await update_operation_state("op1", "restart_pending")

    assert saved["state"].mcp.operations[0].state == "restart_pending"


# ---------------------------------------------------------------------------
# module-level singleton
# ---------------------------------------------------------------------------


def test_manager_singleton_set_and_get():
    manager = MCPManager()
    try:
        set_manager(manager)
        assert get_manager() is manager
    finally:
        set_manager(None)


def test_get_manager_returns_none_before_set():
    original = mcp_manager._manager
    try:
        mcp_manager._manager = None
        assert get_manager() is None
    finally:
        mcp_manager._manager = original
