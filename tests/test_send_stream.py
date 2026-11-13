"""Tests for TelegramInterface.send_stream — sendMessageDraft streaming."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

import aug.api.interfaces.telegram.interface as _iface_mod
from aug.core.harness import TextDelta, ToolEnded, ToolStarted


def _make_context(chat_id: int = 123, context_id: int | None = None):
    """Build a minimal mock Update context for send_stream tests."""
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    bot.send_chat_action = AsyncMock()

    sent: list[MagicMock] = []

    def _make_msg(text, **kwargs):
        m = MagicMock()
        m.text = text
        m.call_kwargs = kwargs
        m.delete = AsyncMock()
        m.edit_text = AsyncMock()
        sent.append(m)
        return m

    msg = MagicMock()
    msg.get_bot.return_value = bot
    msg.chat_id = chat_id
    msg.message_thread_id = context_id
    msg.reply_text = AsyncMock(side_effect=lambda text, **kw: _make_msg(text, **kw))

    context = MagicMock()
    context.effective_message = msg
    context.effective_chat.id = chat_id
    context.get_bot.return_value = bot

    return context, msg, bot, sent


async def _stream(*events):
    for e in events:
        yield e


@pytest.fixture(autouse=True)
def _no_typing_loop():
    with patch(
        "aug.api.interfaces.telegram.interface._typing_loop",
        new=AsyncMock(),
    ):
        yield


@pytest.fixture()
def interface():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    return TelegramInterface()


# ---------------------------------------------------------------------------
# Tracer bullet — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_called_during_stream_and_final_message_commits(interface):
    """send_message_draft is called while streaming; reply_text commits the final result."""
    context, msg, bot, _sent = _make_context()

    await interface.send_stream(
        _stream(
            TextDelta(text="Hello"),
            TextDelta(text=" world"),
        ),
        context,
    )

    assert bot.send_message_draft.called
    # Final committed message was sent
    assert msg.reply_text.called
    # Last reply_text call must NOT be silent (user gets a notification)
    last_kwargs = msg.reply_text.call_args[1]
    assert not last_kwargs.get("disable_notification", False)


# ---------------------------------------------------------------------------
# Throttle — rapid tokens don't spam the draft API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rapid_tokens_throttled_to_one_draft_call(interface):
    """Ten tokens delivered in rapid succession produce only one draft call (300 ms throttle)."""
    context, _msg, bot, _sent = _make_context()

    tokens = [TextDelta(text=c) for c in "Hello world!"]
    await interface.send_stream(_stream(*tokens), context)

    # Only the first token triggers a draft; the rest are throttled
    assert bot.send_message_draft.call_count == 1


# ---------------------------------------------------------------------------
# Tool interlude — text before tool is committed silently; text after commits with notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_interlude_commits_pretext_silently(interface):
    """Text accumulated before a tool starts is sent as a silent committed message."""
    context, msg, _bot, sent = _make_context()

    await interface.send_stream(
        _stream(
            TextDelta(text="Before"),
            ToolStarted(id="r1", name="brave_search", args={"query": "q"}),
            ToolEnded(id="r1", name="brave_search", output=None, error=False),
            TextDelta(text="After"),
        ),
        context,
    )

    # At least two reply_text calls: one silent (pre-tool commit) + one notifying (final)
    assert msg.reply_text.call_count >= 2

    # The pre-tool commit must be silent
    pre_tool_call = sent[0]  # first committed message
    assert pre_tool_call.call_kwargs.get("disable_notification", False)

    # The final message must notify
    last_kwargs = msg.reply_text.call_args[1]
    assert not last_kwargs.get("disable_notification", False)


# ---------------------------------------------------------------------------
# Fallback — when sendMessageDraft fails, log and fall back to editMessageText
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_generic_failure_logs_warning_and_falls_back(interface):
    """Non-peer-invalid errors log a warning and fall back to editMessageText."""
    context, msg, bot, _sent = _make_context()
    bot.send_message_draft = AsyncMock(side_effect=BadRequest("Something went wrong"))

    with patch("aug.api.interfaces.telegram.interface.logger") as mock_logger:
        await interface.send_stream(
            _stream(
                TextDelta(text="Hello"),
                TextDelta(text=" world"),
            ),
            context,
        )

    mock_logger.warning.assert_called_once()
    assert msg.reply_text.call_count >= 1


@pytest.mark.asyncio
async def test_peer_invalid_error_logged_at_info_and_chat_cached(interface):
    """Textdraft_peer_invalid logs at INFO and caches the chat so draft is never tried again."""
    context, _msg, bot, _sent = _make_context(chat_id=999)
    bot.send_message_draft = AsyncMock(side_effect=BadRequest("Textdraft_peer_invalid"))

    try:
        with patch("aug.api.interfaces.telegram.interface.logger") as mock_logger:
            await interface.send_stream(
                _stream(TextDelta(text="Hello")),
                context,
            )

        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called()
        assert 999 in _iface_mod._no_draft_chats

        # Second request: draft never attempted
        bot.send_message_draft.reset_mock()
        await interface.send_stream(_stream(TextDelta(text="Hi")), context)
        bot.send_message_draft.assert_not_called()
    finally:
        _iface_mod._no_draft_chats.discard(999)


# ---------------------------------------------------------------------------
# Empty stream — no crash, nothing sent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_stream_sends_nothing(interface):
    context, msg, bot, _sent = _make_context()

    await interface.send_stream(_stream(), context)

    bot.send_message_draft.assert_not_called()
    msg.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_failed_turn_sends_no_partial_answer_and_marks_open_tools_failed(interface):
    """A turn ending in an error behaves like the old raised exception: nothing more is
    committed (the caller sends the error instead), and a tool still running is frozen
    as failed."""
    from aug.core.harness import HarnessError, TurnEnded

    context, _msg, _bot, sent = _make_context()

    await interface.send_stream(
        _stream(
            ToolStarted(id="r1", name="run_bash", args={"command": "ls"}),
            TurnEnded(error=HarnessError(kind="rate_limited")),
        ),
        context,
    )

    texts = [m.text for m in sent]
    assert len(texts) == 1 and "🔴" in texts[0]  # only the tool line, frozen as failed


@pytest.mark.asyncio
async def test_stopped_turn_marks_its_open_tools_stopped_not_failed(interface):
    from aug.core.harness import TurnEnded

    context, _msg, _bot, sent = _make_context()

    await interface.send_stream(
        _stream(
            ToolStarted(id="r1", name="run_bash", args={"command": "ls"}),
            ToolEnded(id="r1", name="run_bash", output="ok"),
            ToolStarted(id="r2", name="run_bash", args={"command": "sleep 60"}),
            TurnEnded(cancelled=True),
        ),
        context,
    )

    final = sent[-1].text
    assert "🟢" in final and "⏹" in final and "🔴" not in final


# ---------------------------------------------------------------------------
# Nested subagent events and approvals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_tool_calls_nest_under_the_subagent_line(interface):
    from aug.core.harness import ToolProgress

    context, _msg, _bot, sent = _make_context()

    await interface.send_stream(
        _stream(
            ToolStarted(id="r1", name="run_subagent", args={"prompt": "look"}),
            ToolStarted(id="s1", name="brave_search", args={"query": "q"}, parent_id="r1"),
            ToolProgress(text="Step 2 · example.com", parent_id="r1"),
            ToolEnded(id="s1", name="brave_search", output="hits", parent_id="r1"),
            ToolEnded(id="r1", name="run_subagent", output="found"),
            TextDelta(text="Here it is."),
        ),
        context,
    )

    [block] = [m.text for m in sent if "Here it is" not in m.text]
    assert "Step 2 · example.com" in block
    assert block.count("🟢") == 1  # the subagent line; its children have no status of their own


def _approval_events(decision):
    from aug.core.harness import (
        ApprovalRequest,
        ApprovalRequested,
        ApprovalResolved,
    )

    request = ApprovalRequest(tool_name="run_ssh", resource="box", operation="rm x")
    return [
        ToolStarted(id="r1", name="run_ssh", args={"target": "box", "command": "rm x"}),
        ApprovalRequested(id="a1", request=request, parent_id="r1"),
        ApprovalResolved(id="a1", decision=decision, parent_id="r1"),
    ]


@pytest.mark.asyncio
async def test_approval_prompt_carries_request_id_buttons_and_shows_the_answer(interface):
    from aug.core.harness import ApprovalDecision

    context, _msg, _bot, sent = _make_context()

    await interface.send_stream(
        _stream(
            *_approval_events(ApprovalDecision.DENIED),
            ToolEnded(id="r1", name="run_ssh", output="denied", error=True),
            TextDelta(text="Not run."),
        ),
        context,
    )

    prompt = next(m for m in sent if "Approval required" in m.text)
    rows = prompt.call_kwargs["reply_markup"].inline_keyboard
    assert [row[0].callback_data for row in rows] == ["ap:a1:o", "ap:a1:a", "ap:a1:d"]
    prompt.edit_text.assert_awaited_with("❌ Command denied.")
    blocks = [m.text for m in sent if "run_ssh" in m.text.lower() or "🔴" in m.text]
    status = [t for t in blocks if "Approval required" not in t]
    assert len(status) == 1 and "🔴" in status[0]  # one status block; the denied tool is red


@pytest.mark.asyncio
async def test_an_unanswered_prompt_loses_its_buttons_when_the_turn_stops(interface):
    from aug.core.harness import ApprovalRequest, ApprovalRequested, TurnEnded

    context, _msg, _bot, sent = _make_context()
    request = ApprovalRequest(tool_name="run_ssh", resource="box", operation="ls")

    await interface.send_stream(
        _stream(
            ToolStarted(id="r1", name="run_ssh", args={}),
            ApprovalRequested(id="a1", request=request, parent_id="r1"),
            TurnEnded(cancelled=True),
        ),
        context,
    )

    prompt = next(m for m in sent if "Approval required" in m.text)
    prompt.edit_text.assert_awaited_with("⏹ Not run — stopped.")
