"""Tests for TelegramInterface.send_stream — sendMessageDraft streaming."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

import aug.api.interfaces.telegram.interface as _iface_mod
from aug.core.events import ChatModelStreamEvent, ToolEndEvent, ToolStartEvent


def _make_context(chat_id: int = 123, thread_id: int | None = None):
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
    msg.message_thread_id = thread_id
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

    return TelegramInterface(checkpointer=MagicMock())


# ---------------------------------------------------------------------------
# Tracer bullet — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_called_during_stream_and_final_message_commits(interface):
    """send_message_draft is called while streaming; reply_text commits the final result."""
    context, msg, bot, _sent = _make_context()

    await interface.send_stream(
        _stream(
            ChatModelStreamEvent(delta="Hello"),
            ChatModelStreamEvent(delta=" world"),
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

    tokens = [ChatModelStreamEvent(delta=c) for c in "Hello world!"]
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
            ChatModelStreamEvent(delta="Before"),
            ToolStartEvent(
                run_id="r1", tool_name="brave_search", args={"query": "q"}, parent_ids=[]
            ),
            ToolEndEvent(run_id="r1", tool_name="brave_search", output=None, error=False),
            ChatModelStreamEvent(delta="After"),
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
                ChatModelStreamEvent(delta="Hello"),
                ChatModelStreamEvent(delta=" world"),
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
                _stream(ChatModelStreamEvent(delta="Hello")),
                context,
            )

        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called()
        assert 999 in _iface_mod._no_draft_chats

        # Second request: draft never attempted
        bot.send_message_draft.reset_mock()
        await interface.send_stream(_stream(ChatModelStreamEvent(delta="Hi")), context)
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
