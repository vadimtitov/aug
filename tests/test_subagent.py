"""Tests for the subagent tool and BaseAgent.arun().

Behavioral focus:
- arun() yields events from a no-checkpointer graph run
- run_subagent tool accumulates TextDelta text as the return value
- tool returns a graceful error string when the subagent hits its recursion limit
- tool re-emits ToolStarted and ToolProgress under its own call, approvals as they are
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from aug.core.agents.base_agent import BaseAgent
from aug.core.harness import (
    ApprovalRequest,
    ApprovalRequested,
    TextDelta,
    ToolEnded,
    ToolProgress,
    ToolStarted,
)
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.subagent import make_run_subagent_tool

# ---------------------------------------------------------------------------
# Minimal concrete agent — no LLM, no tools, fixed response
# ---------------------------------------------------------------------------


class _FixedAgent(BaseAgent):
    """Returns a fixed AIMessage without calling an LLM.  No tool calls."""

    def __init__(self, response: str = "fixed response") -> None:
        super().__init__()
        self._response = response

    async def respond(self, state: AgentState) -> AgentStateUpdate:
        return AgentStateUpdate(messages=[AIMessage(content=self._response)])


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_fake_arun(*events) -> AsyncIterator:
    """Return an async generator that yields the given events."""

    async def _gen(prompt, *, interface, sender_id, thread_id, tool_context=None):
        for event in events:
            yield event

    return _gen


def _config(thread_id: str = "test-thread") -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "interface": "rest_api",
            "sender_id": "user-1",
        }
    }


# ---------------------------------------------------------------------------
# BaseAgent.arun() — basic contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_completes_on_fixed_agent():
    """arun() runs to completion on a simple agent and yields at least zero events."""
    agent = _FixedAgent()
    events = []
    async for event in agent.arun(
        "hello",
        interface="rest_api",
        sender_id="",
        thread_id="t-1",
    ):
        events.append(event)
    # No assertion on event count — _FixedAgent has no LLM so no stream events.
    # The important thing: no exception, graph ran to END.


@pytest.mark.asyncio
async def test_arun_uses_separate_graph_from_interactive():
    """arun() writes to _subagent_graph and does not touch _compiled_graph."""
    agent = _FixedAgent()

    # Simulate an interactive graph already cached without calling _build()
    sentinel = object()
    agent._compiled_graph = sentinel

    async for _ in agent.arun("hi", interface="rest_api", sender_id="", thread_id="t"):
        pass

    # Interactive graph must be untouched
    assert agent._compiled_graph is sentinel
    # Subagent graph was compiled separately
    assert agent._subagent_graph is not None


@pytest.mark.asyncio
async def test_arun_caches_subagent_graph():
    """_subagent_graph is compiled once and reused on subsequent arun() calls."""
    agent = _FixedAgent()

    async for _ in agent.arun("a", interface="rest_api", sender_id="", thread_id="t"):
        pass
    first = agent._subagent_graph

    async for _ in agent.arun("b", interface="rest_api", sender_id="", thread_id="t"):
        pass

    assert agent._subagent_graph is first


# ---------------------------------------------------------------------------
# run_subagent tool — text accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_accumulates_stream_events():
    """run_subagent returns the accumulated TextDelta text."""
    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    fake_arun = _make_fake_arun(
        TextDelta(text="Hello "),
        TextDelta(text="world"),
    )
    with patch.object(subagent, "arun", fake_arun):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_tool_returns_error_on_recursion_limit():
    """run_subagent returns a graceful error string when GraphRecursionError is raised."""
    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    async def _raising_arun(prompt, *, interface, sender_id, thread_id, tool_context=None):
        raise GraphRecursionError("limit")
        yield  # noqa: unreachable — makes this an async generator

    with patch.object(subagent, "arun", _raising_arun):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert "step limit" in result.lower()


@pytest.mark.asyncio
async def test_tool_returns_partial_text_on_recursion_limit():
    """When the subagent produces some text before hitting the limit, it is included."""
    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    async def _partial_then_error(prompt, *, interface, sender_id, thread_id, tool_context=None):
        yield TextDelta(text="partial answer")
        raise GraphRecursionError("limit")

    with patch.object(subagent, "arun", _partial_then_error):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert "partial answer" in result
    assert "step limit" in result.lower()


# ---------------------------------------------------------------------------
# run_subagent tool — progress forwarding
# ---------------------------------------------------------------------------


def _capturing_context() -> MagicMock:
    return MagicMock(emit=AsyncMock())


@pytest.mark.asyncio
async def test_tool_re_emits_subagent_tool_calls_under_its_own_call():
    """A subagent's ToolStarted is re-emitted with no parent, so the harness nests it
    under this run_subagent call."""
    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)
    ctx = _capturing_context()

    fake_arun = _make_fake_arun(
        ToolStarted(id="r1", name="brave_search", args={"query": "q"}, parent_id="inner"),
        TextDelta(text="done"),
    )
    with (
        patch.object(subagent, "arun", fake_arun),
        patch("aug.core.tools.subagent.current_tool_context", return_value=ctx),
    ):
        await tool.ainvoke({"prompt": "test"}, config=_config())

    ctx.emit.assert_awaited_once_with(
        ToolStarted(id="r1", name="brave_search", args={"query": "q"})
    )


@pytest.mark.asyncio
async def test_tool_forwards_tool_progress_and_approvals():
    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)
    ctx = _capturing_context()
    asked = ApprovalRequested(
        id="a1", request=ApprovalRequest("run_ssh", "box", "ls"), parent_id="inner"
    )

    fake_arun = _make_fake_arun(
        ToolProgress(text="Step 3 · example.com", parent_id="inner"),
        asked,
        ToolEnded(id="s1", name="run_ssh", output="ok", parent_id="inner"),
        TextDelta(text="result"),
    )
    with (
        patch.object(subagent, "arun", fake_arun),
        patch("aug.core.tools.subagent.current_tool_context", return_value=ctx),
    ):
        await tool.ainvoke({"prompt": "test"}, config=_config())

    emitted = [call.args[0] for call in ctx.emit.await_args_list]
    assert emitted == [
        ToolProgress(text="Step 3 · example.com"),
        asked,
        ToolEnded(id="s1", name="run_ssh", output="ok"),
    ]
