"""Tests for Telegram forum topic thread routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.telegram.utils import get_thread_id


def _make_update(chat_id: int, topic_id: int | None) -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.message_thread_id = topic_id
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture(autouse=True)
def allow_all_chats():
    """Bypass TELEGRAM_ALLOWED_CHAT_IDS so handlers run in tests."""
    with patch("aug.api.interfaces.telegram.utils.is_allowed", return_value=True):
        yield


def test_named_topic_produces_topic_thread_id():
    thread_id = get_thread_id(chat_id=123, topic_id=42)
    assert thread_id == "tg-123-topic-42"


def test_same_topic_different_chats_are_isolated():
    assert get_thread_id(chat_id=111, topic_id=1) != get_thread_id(chat_id=222, topic_id=1)


def test_different_topics_same_chat_are_isolated():
    assert get_thread_id(chat_id=123, topic_id=1) != get_thread_id(chat_id=123, topic_id=2)


def test_no_topic_falls_back_to_session_based():
    with patch("aug.api.interfaces.telegram.utils.load_state") as mock_load:
        mock_load.return_value.telegram.chats = {}
        thread_id = get_thread_id(chat_id=123, topic_id=None)
    assert thread_id == "tg-123-0"


def test_no_topic_reflects_non_zero_session():
    from aug.utils.state import AppState, TelegramChatState

    state = AppState()
    state.telegram.chats["123"] = TelegramChatState(session=5)
    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=state):
        thread_id = get_thread_id(chat_id=123, topic_id=None)
    assert thread_id == "tg-123-5"


# ---------------------------------------------------------------------------
# /clear command behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def telegram_interface():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    return TelegramInterface(checkpointer=MagicMock())


@pytest.mark.asyncio
async def test_clear_in_named_topic_replies_with_explanation(telegram_interface):
    update = _make_update(chat_id=123, topic_id=7)
    saved_states: list = []

    with patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved_states.append):
        await telegram_interface._handle_clear(update, MagicMock())

    reply_text = update.message.reply_text.call_args[0][0]
    assert "topic" in reply_text.lower()
    assert not saved_states  # state must NOT be mutated


@pytest.mark.asyncio
async def test_clear_outside_topic_increments_session(telegram_interface):
    from aug.utils.state import AppState, TelegramChatState

    state = AppState()
    state.telegram.chats["123"] = TelegramChatState(session=2)
    saved_states: list = []

    update = _make_update(chat_id=123, topic_id=None)

    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved_states.append),
    ):
        await telegram_interface._handle_clear(update, MagicMock())

    assert len(saved_states) == 1
    assert saved_states[0].telegram.chats["123"].session == 3
