"""Tests for the subagent tool and BaseAgent.arun().

Behavioral focus:
- arun() yields events from a no-checkpointer graph run
- run_subagent tool accumulates ChatModelStreamEvent text as the return value
- tool returns a graceful error string when the subagent hits its recursion limit
- tool forwards ToolStartEvent and ToolProgressEvent as send_tool_progress_update calls
"""

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from aug.core.agents.base_agent import BaseAgent
from aug.core.events import (
    ChatModelStreamEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from aug.core.state import AgentState, AgentStateUpdate

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

    async def _gen(prompt, *, interface, sender_id, thread_id):
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
    """run_subagent returns the accumulated ChatModelStreamEvent text."""
    from aug.core.tools.subagent import make_run_subagent_tool

    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    fake_arun = _make_fake_arun(
        ChatModelStreamEvent(delta="Hello "),
        ChatModelStreamEvent(delta="world"),
    )
    with patch.object(subagent, "arun", fake_arun):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_tool_returns_error_on_recursion_limit():
    """run_subagent returns a graceful error string when GraphRecursionError is raised."""
    from langgraph.errors import GraphRecursionError

    from aug.core.tools.subagent import make_run_subagent_tool

    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    async def _raising_arun(prompt, *, interface, sender_id, thread_id):
        raise GraphRecursionError("limit")
        yield  # noqa: unreachable — makes this an async generator

    with patch.object(subagent, "arun", _raising_arun):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert "step limit" in result.lower()


@pytest.mark.asyncio
async def test_tool_returns_partial_text_on_recursion_limit():
    """When the subagent produces some text before hitting the limit, it is included."""
    from langgraph.errors import GraphRecursionError

    from aug.core.tools.subagent import make_run_subagent_tool

    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    async def _partial_then_error(prompt, *, interface, sender_id, thread_id):
        yield ChatModelStreamEvent(delta="partial answer")
        raise GraphRecursionError("limit")

    with patch.object(subagent, "arun", _partial_then_error):
        result = await tool.ainvoke({"prompt": "test"}, config=_config())

    assert "partial answer" in result
    assert "step limit" in result.lower()


# ---------------------------------------------------------------------------
# run_subagent tool — progress forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_emits_subagent_tool_start_on_tool_start():
    """ToolStartEvent from the subagent triggers send_subagent_tool_start in parent context."""
    from aug.core.tools.subagent import make_run_subagent_tool

    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    fake_arun = _make_fake_arun(
        ToolStartEvent(run_id="r1", tool_name="brave_search", args={"query": "test query"}),
        ChatModelStreamEvent(delta="done"),
    )

    calls: list[tuple[str, dict]] = []

    async def _capture(tool_name: str, args: dict) -> None:
        calls.append((tool_name, args))

    with (
        patch.object(subagent, "arun", fake_arun),
        patch("aug.core.tools.subagent.send_subagent_tool_start", _capture),
    ):
        await tool.ainvoke({"prompt": "test"}, config=_config())

    assert len(calls) == 1
    assert calls[0] == ("brave_search", {"query": "test query"})


@pytest.mark.asyncio
async def test_tool_forwards_tool_progress_events():
    """ToolProgressEvent step text is forwarded verbatim to send_tool_progress_update."""
    from aug.core.tools.subagent import make_run_subagent_tool

    subagent = _FixedAgent()
    tool = make_run_subagent_tool(subagent)

    fake_arun = _make_fake_arun(
        ToolProgressEvent(step="Step 3 · example.com"),
        ChatModelStreamEvent(delta="result"),
    )

    progress_calls: list[str] = []

    async def _capture_progress(text: str) -> None:
        progress_calls.append(text)

    with (
        patch.object(subagent, "arun", fake_arun),
        patch("aug.core.tools.subagent.send_tool_progress_update", _capture_progress),
    ):
        await tool.ainvoke({"prompt": "test"}, config=_config())

    assert "Step 3 · example.com" in progress_calls
