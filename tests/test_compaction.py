"""Tests for context compaction logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from aug.core.compaction import compact_messages, count_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(summary_text: str = "summary of old messages"):
    fake_response = MagicMock()
    fake_response.content = summary_text
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=fake_response)
    return mock_llm


def _research_thread():
    """Realistic thread: one completed turn + current research turn in progress."""
    return [
        # pre-run: completed turn
        HumanMessage(content="what is the capital of France?", id="h0"),
        AIMessage(content="Paris.", id="a0"),
        # current run: last human question + tool calls in progress
        HumanMessage(content="research free speech laws in Europe", id="h1"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "search", "args": {}}], id="a1"),
        ToolMessage(
            content="search result 1: Estonia has no hate speech law", tool_call_id="tc1", id="t1"
        ),
        AIMessage(content="", tool_calls=[{"id": "tc2", "name": "fetch", "args": {}}], id="a2"),
        ToolMessage(
            content="fetch result 2: detailed article on free speech", tool_call_id="tc2", id="t2"
        ),
    ]


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_empty():
    assert count_tokens([]) == 0


def test_count_tokens_positive_for_non_empty():
    assert count_tokens([HumanMessage(content="hello world")]) > 0


def test_count_tokens_grows_with_content():
    assert count_tokens([HumanMessage(content="hi " * 1000)]) > count_tokens(
        [HumanMessage(content="hi")]
    )


# ---------------------------------------------------------------------------
# compact_messages — no-op cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_noop_when_no_human_messages():
    messages = [AIMessage(content="hello", id="a1")]
    result, state_changes = await compact_messages(messages, "model", context_window=200_000)
    assert result is messages
    assert state_changes == []


@pytest.mark.asyncio
async def test_compact_messages_noop_when_no_prerun_and_only_human_message():
    """Only a HumanMessage with no response yet — nothing to summarise."""
    messages = [HumanMessage(content="hi", id="h0")]
    result, state_changes = await compact_messages(messages, "model", context_window=200_000)
    assert result is messages
    assert state_changes == []


@pytest.mark.asyncio
async def test_compact_messages_compacts_current_run_when_no_prerun():
    """No pre-run but current run has tool call history — should compact it.

    Regression: previously no-oped because current_run < 50% of context_window,
    even when the caller's threshold was already exceeded.
    """
    messages = [
        HumanMessage(content="research X", id="h0"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "search", "args": {}}], id="a0"),
        ToolMessage(content="result " * 200, tool_call_id="tc1", id="t0"),
    ]
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm("summary")):
        messages_for_llm, state_changes = await compact_messages(
            messages, "cheap-model", context_window=200_000
        )

    # Tool call history must be summarised
    removed_ids = {m.id for m in state_changes if isinstance(m, RemoveMessage)}
    assert "a0" in removed_ids
    assert "t0" in removed_ids
    # The human message must be kept
    assert "h0" not in removed_ids
    kept_contents = [m.content for m in messages_for_llm]
    assert "research X" in kept_contents


# ---------------------------------------------------------------------------
# compact_messages — pre-run summarised, current run kept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_summarises_prerun_when_current_run_small():
    messages = _research_thread()
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm("France is Paris")):
        messages_for_llm, _state_changes = await compact_messages(
            messages, "cheap-model", context_window=200_000
        )

    # Current run (h1 + tool calls) must be kept verbatim
    contents = [m.content for m in messages_for_llm]
    assert "research free speech laws in Europe" in contents
    assert "search result 1: Estonia has no hate speech law" in contents

    # Pre-run (h0, a0) must NOT appear verbatim
    assert "what is the capital of France?" not in contents
    assert "Paris." not in contents


@pytest.mark.asyncio
async def test_compact_messages_prerun_content_goes_to_llm():
    messages = _research_thread()
    captured = {}

    async def fake_ainvoke(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.content = "summary"
        return r

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    with patch("aug.core.compaction.build_chat_model", return_value=mock_llm):
        await compact_messages(messages, "cheap-model", context_window=200_000)

    assert "what is the capital of France?" in captured["prompt"]
    assert "Paris." in captured["prompt"]
    # current run content must NOT be in summarisation input
    assert "research free speech laws in Europe" not in captured["prompt"]


@pytest.mark.asyncio
async def test_compact_messages_state_changes_remove_prerun_messages():
    messages = _research_thread()
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm()):
        _, state_changes = await compact_messages(messages, "cheap-model", context_window=200_000)

    removed_ids = {m.id for m in state_changes if isinstance(m, RemoveMessage)}
    assert "h0" in removed_ids
    assert "a0" in removed_ids
    # current run messages must NOT be removed
    assert "h1" not in removed_ids
    assert "t1" not in removed_ids


@pytest.mark.asyncio
async def test_compact_messages_summary_is_system_message():
    messages = _research_thread()
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm("France is Paris")):
        messages_for_llm, state_changes = await compact_messages(
            messages, "cheap-model", context_window=200_000
        )

    # First message in llm view must be a SystemMessage containing the summary
    assert isinstance(messages_for_llm[0], SystemMessage)
    assert "France is Paris" in messages_for_llm[0].content

    # state_changes must also include the SystemMessage
    summary_msgs = [m for m in state_changes if isinstance(m, SystemMessage)]
    assert len(summary_msgs) == 1
    assert "France is Paris" in summary_msgs[0].content


# ---------------------------------------------------------------------------
# compact_messages — heavy current run: everything summarised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_summarises_everything_when_current_run_heavy():
    """When current run > 50% of context, tool calls from current run are summarised too."""
    # Make current run tokens > 50% of a tiny context_window
    # context_window=10, current run tool response is "x" * 1000 = 250 tokens >> 5
    heavy_messages = [
        HumanMessage(content="old question", id="h0"),
        AIMessage(content="old answer", id="a0"),
        HumanMessage(content="new question", id="h1"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "fetch", "args": {}}], id="a1"),
        ToolMessage(content="x" * 4000, tool_call_id="tc1", id="t1"),  # huge
    ]
    captured = {}

    async def fake_ainvoke(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.content = "everything summarised"
        return r

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    with patch("aug.core.compaction.build_chat_model", return_value=mock_llm):
        messages_for_llm, state_changes = await compact_messages(
            heavy_messages,
            "cheap-model",
            context_window=10,  # tiny window
        )

    # The huge tool response must be in the summarisation input
    assert "x" * 100 in captured["prompt"]

    # Only the last HumanMessage is kept verbatim
    kept_contents = [m.content for m in messages_for_llm]
    assert "new question" in kept_contents
    # Tool response must NOT be in the llm view
    assert "x" * 100 not in str(kept_contents)

    # The huge ToolMessage must be in state_changes as RemoveMessage
    removed_ids = {m.id for m in state_changes if isinstance(m, RemoveMessage)}
    assert "t1" in removed_ids
    assert "a1" in removed_ids


@pytest.mark.asyncio
async def test_compact_messages_heavy_current_run_keeps_last_human_message():
    heavy_messages = [
        HumanMessage(content="old question", id="h0"),
        AIMessage(content="old answer", id="a0"),
        HumanMessage(content="current question", id="h1"),
        ToolMessage(content="x" * 4000, tool_call_id="tc1", id="t1"),
    ]
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm()):
        messages_for_llm, state_changes = await compact_messages(
            heavy_messages, "cheap-model", context_window=10
        )

    kept_contents = [m.content for m in messages_for_llm]
    assert "current question" in kept_contents

    # last HumanMessage must NOT be in state_changes as RemoveMessage
    removed_ids = {m.id for m in state_changes if isinstance(m, RemoveMessage)}
    assert "h1" not in removed_ids


# ---------------------------------------------------------------------------
# compact_messages — loop guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_injects_loop_guard_on_recompaction():
    """When a previous SystemMessage summary is being re-compacted, the loop guard
    is injected into messages_for_llm (but NOT persisted via state_changes)."""
    messages = [
        # previous summary already in state — first compaction already happened
        SystemMessage(content="[Conversation summary]: previous research...", id="s0"),
        HumanMessage(content="research London schools", id="h0"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "search", "args": {}}], id="a0"),
        ToolMessage(content="x" * 4000, tool_call_id="tc1", id="t0"),
    ]
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm("new summary")):
        messages_for_llm, state_changes = await compact_messages(
            messages, "cheap-model", context_window=10
        )

    # Loop guard must appear in messages_for_llm
    assert any(
        "Do NOT call any more tools" in m.content
        for m in messages_for_llm
        if isinstance(m, SystemMessage)
    )
    # Loop guard must NOT be persisted — not in state_changes
    assert not any(
        "Do NOT call any more tools" in m.content
        for m in state_changes
        if isinstance(m, SystemMessage)
    )


@pytest.mark.asyncio
async def test_compact_messages_no_loop_guard_on_first_compaction():
    """First-time compaction must not inject the loop guard."""
    messages = _research_thread()
    with patch("aug.core.compaction.build_chat_model", return_value=_mock_llm()):
        messages_for_llm, _ = await compact_messages(
            messages, "cheap-model", context_window=200_000
        )

    assert not any(
        "Do NOT call any more tools" in m.content
        for m in messages_for_llm
        if isinstance(m, SystemMessage)
    )


# ---------------------------------------------------------------------------
# compact_messages — max_summary_tokens passed to LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_messages_passes_max_summary_tokens_to_llm():
    messages = _research_thread()
    captured_kwargs = {}

    def fake_build_chat_model(model, **kwargs):
        captured_kwargs.update(kwargs)
        return _mock_llm()

    with patch("aug.core.compaction.build_chat_model", side_effect=fake_build_chat_model):
        await compact_messages(
            messages, "cheap-model", context_window=200_000, max_summary_tokens=123
        )

    assert captured_kwargs.get("max_tokens") == 123


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def _mock_build_chat_model(content="answer"):
    fake_response = AIMessage(content=content)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=fake_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    return MagicMock(return_value=mock_llm)


def _make_aug_agent(**kwargs):
    from aug.core.agents.chat_agent import AugAgent

    with (
        patch("aug.core.agents.chat_agent.build_chat_model", _mock_build_chat_model()),
        patch("aug.core.llm.get_settings") as mock_settings,
    ):
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        return AugAgent(model="gpt-4o", **kwargs)


def _state_with_research():
    from aug.core.state import AgentState

    return AgentState(messages=_research_thread(), thread_id="test", system_prompt="")


@pytest.mark.asyncio
async def test_agent_no_compaction_when_model_none():
    agent = _make_aug_agent(compaction_model=None)
    with (
        patch(
            "aug.core.agents.chat_agent.compact_messages", new_callable=AsyncMock
        ) as mock_compact,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value=""),
    ):
        await agent.respond(_state_with_research())
    mock_compact.assert_not_called()


@pytest.mark.asyncio
async def test_agent_no_compaction_below_threshold():
    agent = _make_aug_agent(compaction_model="cheap", context_window=1_000_000)
    with (
        patch(
            "aug.core.agents.chat_agent.compact_messages", new_callable=AsyncMock
        ) as mock_compact,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value=""),
    ):
        await agent.respond(_state_with_research())
    mock_compact.assert_not_called()


@pytest.mark.asyncio
async def test_agent_compacts_when_threshold_exceeded():
    agent = _make_aug_agent(compaction_model="cheap", context_window=1)
    state = _state_with_research()

    summary_msg = SystemMessage(content="[Summary]: past", id="summary-1")
    remove_msgs = [RemoveMessage(id="h0"), RemoveMessage(id="a0")]
    fake_compacted = _research_thread()[2:]  # current run only

    with (
        patch(
            "aug.core.agents.chat_agent.compact_messages",
            new_callable=AsyncMock,
            return_value=(fake_compacted, [*remove_msgs, summary_msg]),
        ),
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value=""),
    ):
        update = await agent.respond(state)

    returned_types = {type(m).__name__ for m in update.messages}
    assert "RemoveMessage" in returned_types
    assert any(isinstance(m, SystemMessage) and "[Summary]" in m.content for m in update.messages)


@pytest.mark.asyncio
async def test_agent_passes_context_window_and_max_summary_tokens_to_compact():
    agent = _make_aug_agent(
        compaction_model="cheap",
        context_window=1,
        max_summary_tokens=42,
    )
    state = _state_with_research()

    with (
        patch(
            "aug.core.agents.chat_agent.compact_messages",
            new_callable=AsyncMock,
            return_value=(_research_thread(), []),
        ) as mock_compact,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value=""),
    ):
        await agent.respond(state)

    mock_compact.assert_called_once()
    _, kwargs = mock_compact.call_args
    assert kwargs.get("context_window") == 1 or mock_compact.call_args[0][2] == 1


# ---------------------------------------------------------------------------
# compact_thread — on-demand compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_thread_compacts_and_writes_state():
    """compact_thread summarises history and writes state_changes back to the graph."""
    from aug.core.compaction import compact_thread

    agent = _make_aug_agent(compaction_model="cheap", context_window=200_000)

    summary_msg = SystemMessage(content="[Conversation summary]: summary", id="s1")
    remove_msgs = [RemoveMessage(id="h0"), RemoveMessage(id="a0")]
    fake_compacted = _research_thread()[2:]

    mock_checkpointer = MagicMock()
    mock_snapshot = MagicMock()
    mock_snapshot.values = {"messages": _research_thread()}

    with (
        patch.object(agent, "aget_state", new=AsyncMock(return_value=mock_snapshot)),
        patch(
            "aug.core.compaction.compact_messages",
            new=AsyncMock(return_value=(fake_compacted, [*remove_msgs, summary_msg])),
        ),
        patch.object(agent, "_compiled_graph") as mock_graph,
    ):
        mock_graph.aupdate_state = AsyncMock()
        result = await compact_thread(agent, "thread-1", mock_checkpointer)

    assert result is True
    mock_graph.aupdate_state.assert_called_once()
    call_args = mock_graph.aupdate_state.call_args
    written_messages = call_args[0][1]["messages"]
    assert any(isinstance(m, RemoveMessage) for m in written_messages)
    assert any(isinstance(m, SystemMessage) for m in written_messages)


@pytest.mark.asyncio
async def test_compact_thread_raises_when_no_compaction_model():
    from aug.core.compaction import compact_thread

    agent = _make_aug_agent(compaction_model=None)
    with pytest.raises(ValueError, match="no compaction_model"):
        await compact_thread(agent, "thread-1", MagicMock())


@pytest.mark.asyncio
async def test_compact_thread_returns_false_when_nothing_to_compact():
    from aug.core.compaction import compact_thread

    agent = _make_aug_agent(compaction_model="cheap", context_window=200_000)
    mock_snapshot = MagicMock()
    mock_snapshot.values = {"messages": [HumanMessage(content="hi", id="h0")]}

    with (
        patch.object(agent, "aget_state", new=AsyncMock(return_value=mock_snapshot)),
        patch(
            "aug.core.compaction.compact_messages",
            new=AsyncMock(return_value=([], [])),
        ),
    ):
        result = await compact_thread(agent, "thread-1", MagicMock())

    assert result is False
