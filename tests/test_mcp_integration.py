"""Real, non-mocked integration tests for MCP tool support.

Each test spawns an actual subprocess (a tiny inline ``mcp.server.fastmcp``
script) or exercises the real read-modify-write/approval machinery end to
end, rather than mocking the seam the review found broken — these are the
tests that would have caught the P1s a fully-mocked suite missed.
"""

import asyncio
import contextlib
import logging
import os
import socket
import sys
import textwrap
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

import aug.core.tools.mcp as mcp_tools
from aug.core.agents.chat_agent import ChatAgent
from aug.core.mcp_manager import MCPManager, _read_hushed_secret
from aug.core.state import AgentState
from aug.core.tools.approval import ApprovalDecision
from aug.utils.file_settings import ApprovalRule, AppSettings, McpServerConfig, ToolSettings
from aug.utils.mcp_registry import McpRegistryServer
from aug.utils.state import AppState, McpOperation

_APPROVE_ALL = AppSettings(tools=ToolSettings(approvals=[ApprovalRule(pattern=".*")]))
_P_APPROVAL = "aug.core.tools.approval.load_settings"

_ECHO_SERVER = textwrap.dedent("""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("echo")

    @mcp.tool()
    def echo(text: str) -> str:
        return f"echo: {text}"

    mcp.run(transport="stdio")
""")

_SLOW_SERVER = textwrap.dedent("""
    import time
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("slow")

    @mcp.tool()
    def slow_echo(text: str) -> str:
        time.sleep(5)
        return f"echo: {text}"

    mcp.run(transport="stdio")
""")

_SECRET_LEAKING_SERVER = textwrap.dedent("""
    import os, sys
    from mcp.server.fastmcp import FastMCP

    print(f"booting with token {os.environ.get('MY_SECRET_VAR', '')}", file=sys.stderr, flush=True)

    mcp = FastMCP("secret")

    @mcp.tool()
    def ping() -> str:
        return "pong"

    mcp.run(transport="stdio")
""")

_SECRET_ECHOING_SERVER = textwrap.dedent("""
    import os
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("secret-echo")

    @mcp.tool()
    def whoami() -> str:
        return f"token is {os.environ.get('MY_SECRET_VAR', '')}"

    @mcp.tool()
    def boom() -> str:
        raise RuntimeError(f"upstream rejected token {os.environ.get('MY_SECRET_VAR', '')}")

    mcp.run(transport="stdio")
""")

_HTTP_ECHO_SERVER = textwrap.dedent("""
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("http-echo", port=int(sys.argv[1]))

    @mcp.tool()
    def echo(text: str) -> str:
        return f"echo: {text}"

    mcp.run(transport="streamable-http")
""")


def _stdio_cfg(
    name: str, script: str, tmp_path, env: dict[str, str] | None = None
) -> McpServerConfig:
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(script)
    return McpServerConfig(
        name=name, transport="stdio", command=sys.executable, args=[str(script_path)], env=env or {}
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_port(port: int, timeout_seconds: float = 8.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.1)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                await asyncio.sleep(0.05)
    raise TimeoutError(f"port {port} never opened")


def _server(name: str, required_inputs: list[str] | None = None) -> McpRegistryServer:
    return McpRegistryServer(
        name=name,
        namespace=name.rsplit("/", 1)[0],
        description="test",
        version="1.0.0",
        transport="stdio",
        command=sys.executable,
        args=["-c", "pass"],
        required_inputs=required_inputs or [],
    )


# ---------------------------------------------------------------------------
# 1. Real stdio MCP session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_stdio_session_connect_call_and_clean_shutdown(tmp_path):
    cfg = _stdio_cfg("echo", _ECHO_SERVER, tmp_path)
    manager = MCPManager()
    try:
        await manager._load_one(cfg)
        assert manager.health["echo"].status == "active"
        assert [t.name for t in manager.tools] == ["echo__echo"]

        content, _artifact = await manager.tools[0].coroutine(text="hi")
        assert content[0]["text"] == "echo: hi"
    finally:
        await manager.aclose()  # must not raise — same task enters and exits


# ---------------------------------------------------------------------------
# 2. Real hushed secret resolution past file descriptor 9
# ---------------------------------------------------------------------------


def test_hushed_secret_resolves_past_descriptor_9(tmp_path, monkeypatch):
    """Reproduces the review's exact failure mode: `sh -c '... >&{fd}'` breaks
    past descriptor 9 on Debian's dash. This never touches a shell — a fake
    `hushed` on PATH execs straight through to the real reader, and we force
    the allocated fd well past 9 before calling it."""
    monkeypatch.setenv("MY_TOKEN", "actual-secret-value")
    fake_hushed = tmp_path / "hushed"
    fake_hushed.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    fake_hushed.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    padding = [open(os.devnull) for _ in range(12)]
    try:
        value = _read_hushed_secret("MY_TOKEN")
    finally:
        for f in padding:
            f.close()

    assert value == "actual-secret-value"


# ---------------------------------------------------------------------------
# 3. Approval interrupt/resume — install plan survives across conversations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_plan_survives_a_concurrent_search_in_another_conversation():
    """Conversation A starts installing while conversation B searches in
    between — A resolving "#1" later must still install what A originally
    saw, never B's results."""
    cfg_a = {"configurable": {"thread_id": "conv-a"}}
    cfg_b = {"configurable": {"thread_id": "conv-b"}}
    state = AppState()
    settings = AppSettings(mcp_servers=[])
    saved = []

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_state", return_value=state),
        patch("aug.core.tools.mcp.save_state"),
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.utils.file_settings.load_settings", return_value=settings),
        patch("aug.utils.file_settings.save_settings", side_effect=lambda s: saved.append(s)),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp._schedule_restart"),
    ):
        with patch(
            "aug.core.tools.mcp.McpRegistryClient.search",
            AsyncMock(return_value=[_server("io.github.x/server-postgres")]),
        ):
            await mcp_tools.search_mcp_servers.ainvoke({"query": "postgres"}, config=cfg_a)

        with patch(
            "aug.core.tools.mcp.McpRegistryClient.search",
            AsyncMock(return_value=[_server("io.github.x/server-mysql")]),
        ):
            await mcp_tools.search_mcp_servers.ainvoke({"query": "mysql"}, config=cfg_b)

        output = await mcp_tools.install_mcp_server.ainvoke({"index": 1}, config=cfg_a)

    assert "postgres" in output.lower()
    assert saved[0].mcp_servers[0].name == "postgres-0650d2"


# ---------------------------------------------------------------------------
# 4. Concurrent settings mutations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_installs_of_different_servers_both_persist():
    """Two installs racing through the same serialized read-modify-write must
    both land — neither silently overwrites the other's save."""
    store = {"settings": AppSettings(mcp_servers=[])}
    mcp_tools._state.last_search["conv-a"] = [_server("io.github.x/server-postgres")]
    mcp_tools._state.last_search["conv-b"] = [_server("io.github.x/server-mysql")]

    with (
        patch(_P_APPROVAL, return_value=_APPROVE_ALL),
        patch("aug.core.tools.mcp.load_settings", side_effect=lambda: store["settings"]),
        patch("aug.utils.file_settings.load_settings", side_effect=lambda: store["settings"]),
        patch(
            "aug.utils.file_settings.save_settings",
            side_effect=lambda s: store.__setitem__("settings", s),
        ),
        patch("aug.core.tools.mcp.load_state", return_value=AppState()),
        patch("aug.core.tools.mcp.save_state"),
        patch("aug.utils.state.load_state", return_value=AppState()),
        patch("aug.utils.state.save_state"),
        patch("aug.core.tools.mcp._list_hushed_secrets", return_value=set()),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp._schedule_restart"),
    ):
        await asyncio.gather(
            mcp_tools.install_mcp_server.ainvoke(
                {"index": 1}, config={"configurable": {"thread_id": "conv-a"}}
            ),
            mcp_tools.install_mcp_server.ainvoke(
                {"index": 1}, config={"configurable": {"thread_id": "conv-b"}}
            ),
        )

    names = {s.name for s in store["settings"].mcp_servers}
    assert names == {"postgres-0650d2", "mysql-0650d2"}


# ---------------------------------------------------------------------------
# 5. Timeout handling — graceful tool error, not an escaping exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_mcp_server_times_out_as_tool_error_not_exception(tmp_path):
    cfg = _stdio_cfg("slow", _SLOW_SERVER, tmp_path)
    manager = MCPManager()
    try:
        await manager._load_one(cfg)
        assert manager.health["slow"].status == "active"

        with patch("aug.core.mcp_manager._TOOL_CALL_TIMEOUT", 0.2):
            with pytest.raises(ToolException, match="did NOT complete"):
                await manager.tools[0].coroutine(text="hi")
    finally:
        await manager.aclose()


# ---------------------------------------------------------------------------
# 6. Removal reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_removed_server_reconciles_as_success():
    manager = MCPManager()  # nothing connected — the server is gone, as intended
    state = AppState()
    state.mcp.operations.append(
        McpOperation(id="op1", action="remove", server_name="sentry", state="restart_pending")
    )

    with (
        patch("aug.core.mcp_manager.load_state", return_value=state),
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
    ):
        outcomes = await manager.reconcile_operations()

    assert len(outcomes) == 1
    assert "succeeded" in outcomes[0].summary


# ---------------------------------------------------------------------------
# 7. Secret redaction from real subprocess stderr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_never_appears_in_logs_from_real_subprocess_stderr(tmp_path, caplog):
    cfg = _stdio_cfg(
        "secret", _SECRET_LEAKING_SERVER, tmp_path, env={"MY_SECRET_VAR": "hushed:MY_SECRET_VAR"}
    )
    manager = MCPManager()

    with patch("aug.core.mcp_manager._read_hushed_secret", return_value="super-secret-token-xyz"):
        try:
            with caplog.at_level(logging.INFO, logger="aug.core.mcp_manager"):
                await manager._load_one(cfg)
                await asyncio.sleep(0.2)  # give the stderr pump a moment to log the line
        finally:
            await manager.aclose()

    assert manager.health["secret"].status == "active"
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "super-secret-token-xyz" not in log_text
    assert "[REDACTED]" in log_text


# ---------------------------------------------------------------------------
# 8. End-to-end agent run — fake LLM triggers a real MCP tool call
# ---------------------------------------------------------------------------


class _FakeToolCallingChatModel(BaseChatModel):
    """Returns pre-scripted responses in order — the first requests a tool
    call, the second is the final answer. A minimal stand-in for the real
    LiteLLM-backed model so the agent loop runs unmodified."""

    responses: list[AIMessage]
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"


@pytest.mark.asyncio
async def test_agent_graph_executes_real_mcp_tool_end_to_end(tmp_path):
    """A fake LLM requests the real echo__echo MCP tool; the compiled
    LangGraph loop must execute it for real and feed its result back for the
    fake model's second turn to see and respond to."""
    cfg = _stdio_cfg("echo", _ECHO_SERVER, tmp_path)
    manager = MCPManager()
    await manager._load_one(cfg)
    assert manager.tools, "MCP server failed to connect"

    fake_llm = _FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "echo__echo", "args": {"text": "hi"}, "id": "call_1"}],
            ),
            AIMessage(content="The tool replied: echo: hi"),
        ]
    )

    try:
        with patch("aug.core.agents.chat_agent.build_chat_model", return_value=fake_llm):
            agent = ChatAgent(model="fake-model", tools=manager.tools)

        graph = agent._build_subagent()
        state = AgentState(
            messages=[HumanMessage(content="say hi via the echo tool")],
            thread_id="t1",
            interface="rest_api",
        )
        result = await graph.ainvoke(state, config={"recursion_limit": 10})
    finally:
        await manager.aclose()

    final_message = result["messages"][-1]
    assert "echo: hi" in final_message.content

    tool_message = next(m for m in result["messages"] if getattr(m, "name", None) == "echo__echo")
    assert tool_message.content[0]["text"] == "echo: hi"


# ---------------------------------------------------------------------------
# 9. Real streamable-HTTP MCP session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_http_session_connect_call_and_clean_shutdown(tmp_path):
    """Every other test in this file uses stdio — streamable-HTTP is a
    separate code path in _open_session (headers instead of env/stdin) that
    had no real-transport coverage at all."""
    port = _free_port()
    script_path = tmp_path / "http_echo.py"
    script_path.write_text(_HTTP_ECHO_SERVER)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script_path), str(port), stderr=asyncio.subprocess.DEVNULL
    )
    manager = MCPManager()
    try:
        await _wait_for_port(port)
        cfg = McpServerConfig(name="httpecho", transport="http", url=f"http://127.0.0.1:{port}/mcp")
        await manager._load_one(cfg)
        assert manager.health["httpecho"].status == "active"
        assert [t.name for t in manager.tools] == ["httpecho__echo"]

        content, _artifact = await manager.tools[0].coroutine(text="hi")
        assert content[0]["text"] == "echo: hi"
    finally:
        await manager.aclose()  # must not raise — same task enters and exits
        proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=3)


# ---------------------------------------------------------------------------
# 10. Approval interrupt/resume through a real LangGraph checkpoint
# ---------------------------------------------------------------------------


class _ApprovalGraphState(TypedDict):
    result: str


async def _call_remove_mcp_server(state: _ApprovalGraphState) -> dict:
    result = await mcp_tools.remove_mcp_server.ainvoke({"name": "postgres"})
    return {"result": result}


@pytest.mark.asyncio
async def test_mcp_tool_approval_survives_a_real_checkpoint_interrupt_and_resume():
    """Unlike the plan-persistence tests above (which call tools directly
    with approval always pre-granted), this drives the actual LangGraph
    interrupt: the graph must genuinely pause, park state in a real
    checkpointer, and only run the tool once resumed with a decision."""
    graph = StateGraph(_ApprovalGraphState)
    graph.add_node("call", _call_remove_mcp_server)
    graph.add_edge(START, "call")
    graph.add_edge("call", END)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "approval-checkpoint-t1"}}

    settings = AppSettings(
        mcp_servers=[McpServerConfig(name="postgres", transport="stdio", command="npx", args=[])]
    )
    saved = []

    with (
        patch(_P_APPROVAL, return_value=AppSettings()),  # no saved rule -> must interrupt
        patch("aug.core.tools.mcp.load_settings", return_value=settings),
        patch("aug.utils.file_settings.load_settings", return_value=settings),
        patch("aug.utils.file_settings.save_settings", side_effect=lambda s: saved.append(s)),
        patch("aug.core.tools.mcp.record_operation", AsyncMock(return_value="op1")),
        patch("aug.core.tools.mcp._schedule_restart"),
    ):
        await compiled.ainvoke({"result": ""}, config)
        paused = await compiled.aget_state(config)
        assert paused.interrupts, "graph must actually pause for approval, not just call the tool"
        assert paused.interrupts[0].value.tool_name == "remove_mcp_server"

        final = await compiled.ainvoke(Command(resume=ApprovalDecision.APPROVED_ONCE), config)

    assert "removed" in final["result"].lower()
    assert saved[0].mcp_servers == []


# ---------------------------------------------------------------------------
# 11. Graph-level timeout — the compiled graph completes, it doesn't hang
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_graph_completes_when_an_mcp_tool_call_times_out(tmp_path):
    """test_slow_mcp_server_times_out_as_tool_error_not_exception (above)
    only proves the tool wrapper itself turns a timeout into a ToolException.
    This drives the same slow server through the full compiled agent graph,
    the way a real request would, and checks the graph run finishes on its
    own — a timeout that instead propagated as a bare exception would abort
    ainvoke() entirely rather than handing the model an error ToolMessage."""
    cfg = _stdio_cfg("slow", _SLOW_SERVER, tmp_path)
    manager = MCPManager()
    await manager._load_one(cfg)
    assert manager.tools, "MCP server failed to connect"

    fake_llm = _FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "slow__slow_echo", "args": {"text": "hi"}, "id": "call_1"}],
            ),
            AIMessage(content="The tool timed out."),
        ]
    )

    try:
        with (
            patch("aug.core.agents.chat_agent.build_chat_model", return_value=fake_llm),
            patch("aug.core.mcp_manager._TOOL_CALL_TIMEOUT", 0.2),
        ):
            agent = ChatAgent(model="fake-model", tools=manager.tools)
            graph = agent._build_subagent()
            state = AgentState(
                messages=[HumanMessage(content="run it")], thread_id="t1", interface="rest_api"
            )
            result = await asyncio.wait_for(
                graph.ainvoke(state, config={"recursion_limit": 10}), timeout=5
            )
    finally:
        await manager.aclose()

    assert result["messages"][-1].content == "The tool timed out."
    tool_message = next(
        m for m in result["messages"] if getattr(m, "name", None) == "slow__slow_echo"
    )
    assert "did not complete" in tool_message.content.lower()


# ---------------------------------------------------------------------------
# 12. Secret redaction in successful tool results and tool-execution errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_tool_result_redacts_a_resolved_secret(tmp_path):
    """Item 3: redaction previously only covered stderr and error strings —
    a resolved secret echoed back in a *successful* tool result (e.g. a
    server confirming the token it received) reached the agent verbatim."""
    cfg = _stdio_cfg(
        "secretecho",
        _SECRET_ECHOING_SERVER,
        tmp_path,
        env={"MY_SECRET_VAR": "hushed:MY_SECRET_VAR"},
    )
    manager = MCPManager()
    with patch("aug.core.mcp_manager._read_hushed_secret", return_value="super-secret-token-xyz"):
        await manager._load_one(cfg)
        try:
            whoami = next(t for t in manager.tools if t.name == "secretecho__whoami")
            content, _artifact = await whoami.coroutine()
        finally:
            await manager.aclose()

    text = content[0]["text"]
    assert "super-secret-token-xyz" not in text
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_tool_execution_error_redacts_a_resolved_secret(tmp_path):
    """The same resolved secret, this time surfacing through the server's
    own error path (CallToolResult(isError=True)) rather than a successful
    result or a transport-level exception."""
    cfg = _stdio_cfg(
        "secretecho",
        _SECRET_ECHOING_SERVER,
        tmp_path,
        env={"MY_SECRET_VAR": "hushed:MY_SECRET_VAR"},
    )
    manager = MCPManager()
    with patch("aug.core.mcp_manager._read_hushed_secret", return_value="super-secret-token-xyz"):
        await manager._load_one(cfg)
        try:
            boom = next(t for t in manager.tools if t.name == "secretecho__boom")
            with pytest.raises(ToolException) as excinfo:
                await boom.coroutine()
        finally:
            await manager.aclose()

    assert "super-secret-token-xyz" not in str(excinfo.value)
