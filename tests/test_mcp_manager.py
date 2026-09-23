"""Tests for aug/core/mcp_manager.py — MCPManager, secret resolution, namespacing."""

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aug.core import mcp_manager
from aug.core.mcp_manager import (
    MCPManager,
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
    have written to, mimicking the real `sh -c 'printf ... >&FD'` invocation."""

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
    """A secret name is interpolated into a shell command — it must never reach
    the shell unescaped, regardless of where it came from (settings.json, a
    registry listing, a hand edit)."""
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


@pytest.mark.asyncio
async def test_resolve_hushed_ref_rejects_non_hushed_prefix():
    with pytest.raises(McpSecretError, match="hushed:KEY"):
        await _resolve_hushed_ref("plaintext-value")


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
async def test_namespace_tool_enforces_timeout():
    async def _hangs(text: str = "") -> str:
        await asyncio.sleep(10)
        return "never"

    original = _make_tool("slow", _hangs)
    wrapped = _namespace_tool(original, "server__slow")

    with patch("aug.core.mcp_manager._TOOL_CALL_TIMEOUT", 0.05):
        with pytest.raises(TimeoutError):
            await wrapped.coroutine(text="x")


# ---------------------------------------------------------------------------
# MCPManager.load_all
# ---------------------------------------------------------------------------


def _stdio_cfg(name: str, enabled: bool = True) -> McpServerConfig:
    return McpServerConfig(
        name=name, transport="stdio", command="npx", args=["-y", f"{name}@1.0.0"], enabled=enabled
    )


class _FakeSession:
    def __init__(self):
        self.initialize = AsyncMock()


class _FakeSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


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
async def test_load_all_isolates_per_server_failures():
    settings = AppSettings(mcp_servers=[_stdio_cfg("good"), _stdio_cfg("bad")])

    async def _connect(cfg):
        if cfg.name == "bad":
            raise RuntimeError("connection refused")
        manager.tools.append(_fake_tool(f"{cfg.name}__search"))
        manager.health[cfg.name] = McpServerHealth(cfg.name, cfg.transport, "active", tool_count=1)

    with patch("aug.core.mcp_manager.load_settings", return_value=settings):
        manager = MCPManager()
        with patch.object(manager, "_connect", side_effect=_connect):
            await manager.load_all()

    assert manager.health["good"].status == "active"
    assert manager.health["bad"].status == "failed"
    assert "connection refused" in manager.health["bad"].error
    assert len(manager.tools) == 1


@pytest.mark.asyncio
async def test_load_all_respects_overall_deadline():
    settings = AppSettings(mcp_servers=[_stdio_cfg("slow")])

    async def _connect(cfg):
        await asyncio.sleep(10)

    with patch("aug.core.mcp_manager.load_settings", return_value=settings):
        manager = MCPManager()
        with (
            patch.object(manager, "_connect", side_effect=_connect),
            patch("aug.core.mcp_manager._OVERALL_TIMEOUT", 0.05),
            patch("aug.core.mcp_manager._PER_SERVER_TIMEOUT", 5),
        ):
            await manager.load_all()  # must not raise

    assert manager.tools == []


@pytest.mark.asyncio
async def test_connect_namespaces_tools_and_detects_collision():
    cfg_a = _stdio_cfg("github")

    async def _fake_load_mcp_tools(session, server_name, tool_name_prefix):
        return [_fake_tool("search")]

    manager = MCPManager()
    manager._tool_names = {"github__search"}  # simulate a pre-existing collision

    with (
        patch("aug.core.mcp_manager.create_session", return_value=_FakeSessionCM(_FakeSession())),
        patch("aug.core.mcp_manager.load_mcp_tools", side_effect=_fake_load_mcp_tools),
        patch("aug.core.mcp_manager._resolve_stdio_env", return_value={}),
    ):
        await manager._connect(cfg_a)

    # "github__search" was already reserved, so it's skipped, not added twice.
    assert manager.health["github"].tool_count == 0
    assert manager.tools == []


@pytest.mark.asyncio
async def test_connect_success_namespaces_and_registers_tool():
    cfg = _stdio_cfg("github")

    async def _fake_load_mcp_tools(session, server_name, tool_name_prefix):
        return [_fake_tool("search")]

    manager = MCPManager()
    with (
        patch("aug.core.mcp_manager.create_session", return_value=_FakeSessionCM(_FakeSession())),
        patch("aug.core.mcp_manager.load_mcp_tools", side_effect=_fake_load_mcp_tools),
        patch("aug.core.mcp_manager._resolve_stdio_env", return_value={}),
    ):
        await manager._connect(cfg)

    assert manager.health["github"].status == "active"
    assert manager.health["github"].tool_count == 1
    assert manager.tools[0].name == "github__search"


@pytest.mark.asyncio
async def test_load_one_wraps_secret_error_as_failed_health():
    cfg = _stdio_cfg("needs-secret")
    manager = MCPManager()
    with patch.object(manager, "_connect", side_effect=McpSecretError("secret 'X' is not set")):
        await manager._load_one(cfg)
    assert manager.health["needs-secret"].status == "failed"
    assert "not set" in manager.health["needs-secret"].error


# ---------------------------------------------------------------------------
# reconcile_operations
# ---------------------------------------------------------------------------


def _state_with_pending(server_name: str = "postgres") -> AppState:
    st = AppState()
    st.mcp.operations.append(
        McpOperation(id="op1", action="install", server_name=server_name, state="restart_pending")
    )
    return st


def test_reconcile_operations_marks_active_when_healthy():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth("postgres", "stdio", "active", tool_count=3)

    saved = {}
    with (
        patch("aug.core.mcp_manager.load_state", return_value=_state_with_pending()),
        patch("aug.core.mcp_manager.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        report = manager.reconcile_operations()

    assert "succeeded" in report
    assert saved["state"].mcp.operations[0].state == "active"


def test_reconcile_operations_marks_failed_when_unhealthy():
    manager = MCPManager()
    manager.health["postgres"] = McpServerHealth(
        "postgres", "stdio", "failed", error="connection refused"
    )

    saved = {}
    with (
        patch("aug.core.mcp_manager.load_state", return_value=_state_with_pending()),
        patch("aug.core.mcp_manager.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        report = manager.reconcile_operations()

    assert "failed" in report
    assert "connection refused" in report
    assert saved["state"].mcp.operations[0].state == "failed"


def test_reconcile_operations_missing_server_marks_failed():
    """The server isn't in health at all — e.g. the restart never actually happened."""
    manager = MCPManager()

    saved = {}
    with (
        patch("aug.core.mcp_manager.load_state", return_value=_state_with_pending()),
        patch("aug.core.mcp_manager.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        report = manager.reconcile_operations()

    assert "failed" in report
    assert saved["state"].mcp.operations[0].state == "failed"


def test_reconcile_operations_returns_none_when_nothing_pending():
    manager = MCPManager()
    with patch("aug.core.mcp_manager.load_state", return_value=AppState()):
        assert manager.reconcile_operations() is None


# ---------------------------------------------------------------------------
# record_operation / update_operation_state
# ---------------------------------------------------------------------------


def test_record_operation_persists_and_returns_id():
    saved = {}
    with (
        patch("aug.core.mcp_manager.load_state", return_value=AppState()),
        patch("aug.core.mcp_manager.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        op_id = record_operation("install", "postgres", "saved")

    assert op_id
    op = saved["state"].mcp.operations[0]
    assert op.id == op_id
    assert op.action == "install"
    assert op.server_name == "postgres"
    assert op.state == "saved"


def test_update_operation_state_updates_matching_record():
    st = AppState()
    st.mcp.operations.append(
        McpOperation(id="op1", action="install", server_name="postgres", state="saved")
    )
    saved = {}
    with (
        patch("aug.core.mcp_manager.load_state", return_value=st),
        patch("aug.core.mcp_manager.save_state", side_effect=lambda s: saved.update(state=s)),
    ):
        update_operation_state("op1", "restart_pending")

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
