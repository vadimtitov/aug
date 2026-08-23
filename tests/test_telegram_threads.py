"""Tests for Telegram forum topic thread routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.telegram.utils import get_conversation_id, get_thread_id
from aug.utils.file_settings import AppSettings, ConversationSettings


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


# ---------------------------------------------------------------------------
# Per-conversation agent version (/version)
# ---------------------------------------------------------------------------


def test_conversation_id_strips_session_from_chat_thread():
    assert get_conversation_id("tg-123-5") == "tg-123"


def test_conversation_id_keeps_topic_thread_intact():
    assert get_conversation_id("tg-123-topic-42") == "tg-123-topic-42"


def test_conversation_id_handles_negative_chat_ids():
    assert get_conversation_id("tg--1003820312204-0") == "tg--1003820312204"


def test_conversation_survives_clear(telegram_interface):
    """A new session (thread ID) must keep the version the user picked."""
    assert telegram_interface.conversation_id("tg-123-0") == telegram_interface.conversation_id(
        "tg-123-9"
    )


def _make_callback_update(chat_id: int, topic_id: int | None, agent: str) -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_message.message_thread_id = topic_id
    update.callback_query.data = f"version:{agent}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


@pytest.mark.asyncio
async def test_version_selection_is_scoped_to_one_topic(telegram_interface):
    """Selecting a version in a forum topic must not change its sibling topics."""
    from aug.core.registry import list_agents
    from aug.utils.file_settings import AppSettings

    agent = next(a for a in list_agents() if a != "fake")
    settings = AppSettings()

    with (
        patch("aug.api.interfaces.base.load_settings", return_value=settings),
        patch("aug.api.interfaces.base.save_settings"),
    ):
        await telegram_interface._handle_version_callback(
            _make_callback_update(chat_id=-100, topic_id=7, agent=agent), MagicMock()
        )
        assert telegram_interface.get_agent_version("tg--100-topic-7") == agent
        assert telegram_interface.get_agent_version("tg--100-topic-8") == "default"
        assert telegram_interface.get_agent_version("tg--100-0") == "default"


@pytest.mark.asyncio
async def test_version_selection_persists_across_clear(telegram_interface):
    """A version picked in a DM stays selected after /clear rotates the thread ID."""
    from aug.core.registry import list_agents
    from aug.utils.file_settings import AppSettings

    agent = next(a for a in list_agents() if a != "fake")
    settings = AppSettings()

    with (
        patch("aug.api.interfaces.base.load_settings", return_value=settings),
        patch("aug.api.interfaces.base.save_settings"),
        patch("aug.api.interfaces.telegram.utils.load_state") as mock_state,
    ):
        mock_state.return_value.telegram.chats = {}
        await telegram_interface._handle_version_callback(
            _make_callback_update(chat_id=123, topic_id=None, agent=agent), MagicMock()
        )
        assert telegram_interface.get_agent_version("tg-123-4") == agent


@pytest.mark.asyncio
async def test_version_end_to_end_through_settings_file(telegram_interface, tmp_path):
    """Full path with no settings mocking: pick in one topic, read back in both.

    Drives the real callback handler and the real receive_message against a real
    settings.json on disk, so the whole chain — handler, conversation key, file
    round-trip, agent resolution — is exercised as it runs in production.
    """
    from aug.core.registry import list_agents

    agents = [a for a in list_agents() if a != "fake"]
    agent_a, agent_b = agents[0], agents[1]

    def _message_update(chat_id: int, topic_id: int | None) -> MagicMock:
        update = MagicMock()
        update.effective_chat.id = chat_id
        update.effective_user.id = 1
        update.message.message_thread_id = topic_id
        update.message.voice = None
        update.message.photo = []
        update.message.audio = None
        update.message.sticker = None
        update.message.document = None
        update.message.location = None
        update.message.text = "hi"
        update.message.caption = None
        update.message.forward_origin = None
        return update

    with (
        patch("aug.utils.data.DATA_DIR", tmp_path),
        patch("aug.api.interfaces.telegram.interface.is_allowed", return_value=True),
    ):
        # Pick agent_a in topic 7, agent_b in topic 8 — same group chat.
        await telegram_interface._handle_version_callback(
            _make_callback_update(chat_id=-100, topic_id=7, agent=agent_a), MagicMock()
        )
        await telegram_interface._handle_version_callback(
            _make_callback_update(chat_id=-100, topic_id=8, agent=agent_b), MagicMock()
        )

        # Both selections must survive the file round-trip, independently.
        incoming_7 = await telegram_interface.receive_message(_message_update(-100, 7))
        incoming_8 = await telegram_interface.receive_message(_message_update(-100, 8))

        assert incoming_7.agent_version == agent_a
        assert incoming_8.agent_version == agent_b
        assert agent_a != agent_b  # guards against a vacuous pass

        written = json.loads((tmp_path / "settings.json").read_text())
        assert written["conversations"]["tg--100-topic-7"]["agent"] == agent_a
        assert written["conversations"]["tg--100-topic-8"]["agent"] == agent_b


@pytest.mark.asyncio
async def test_version_callback_refuses_inaccessible_message(telegram_interface):
    """A button pressed on a message the bot can no longer access (~48h old).

    Built from real python-telegram-bot objects, not MagicMocks: an InaccessibleMessage
    carries no message_thread_id, so the topic is genuinely unknown. The handler must
    refuse rather than guess — guessing writes the selection to the wrong conversation,
    which is the exact leak this feature exists to prevent. edit_message_text is left
    unpatched on purpose: it raises TypeError on an inaccessible message, so if the
    handler still tries to edit, this test fails.
    """
    from telegram import CallbackQuery, Chat, InaccessibleMessage, Update, User

    from aug.core.registry import list_agents

    agent = next(a for a in list_agents() if a != "fake")
    chat = Chat(id=-100, type=Chat.SUPERGROUP)
    query = CallbackQuery(
        id="q1",
        from_user=User(id=1, first_name="V", is_bot=False),
        chat_instance="ci",
        data=f"version:{agent}",
        message=InaccessibleMessage(chat=chat, message_id=5),
    )
    update = Update(update_id=1, callback_query=query)
    settings = AppSettings()

    with (
        patch("aug.api.interfaces.base.load_settings", return_value=settings),
        patch("aug.api.interfaces.base.save_settings") as mock_save,
        patch.object(CallbackQuery, "answer", new_callable=AsyncMock) as mock_answer,
    ):
        await telegram_interface._handle_version_callback(update, MagicMock())

    # Exactly once: Telegram rejects a second answer for the same callback query, so a
    # double-answer would raise in production and swallow the alert.
    mock_answer.assert_awaited_once()
    assert mock_answer.await_args.kwargs.get("show_alert") is True
    assert settings.conversations == {}  # nothing guessed, nothing written
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Topic inherits its group's version until one is picked in the topic
# ---------------------------------------------------------------------------


def test_topic_inherits_group_version(telegram_interface):
    """A topic with no selection of its own uses the group's."""
    settings = AppSettings(conversations={"tg--100": ConversationSettings(agent="v9_claude")})
    with patch("aug.api.interfaces.base.load_settings", return_value=settings):
        assert telegram_interface.get_agent_version("tg--100-topic-7") == "v9_claude"


def test_topic_own_version_wins_over_group(telegram_interface):
    settings = AppSettings(
        conversations={
            "tg--100": ConversationSettings(agent="v9_claude"),
            "tg--100-topic-7": ConversationSettings(agent="v11_glm5"),
        }
    )
    with patch("aug.api.interfaces.base.load_settings", return_value=settings):
        assert telegram_interface.get_agent_version("tg--100-topic-7") == "v11_glm5"
        assert telegram_interface.get_agent_version("tg--100-topic-8") == "v9_claude"


def test_setting_a_topic_version_does_not_touch_the_group(telegram_interface):
    settings = AppSettings(conversations={"tg--100": ConversationSettings(agent="v9_claude")})
    with (
        patch("aug.api.interfaces.base.load_settings", return_value=settings),
        patch("aug.api.interfaces.base.save_settings"),
    ):
        telegram_interface.set_agent_version("tg--100-topic-7", "v11_glm5")
    assert settings.conversations["tg--100"].agent == "v9_claude"
    assert settings.conversations["tg--100-topic-7"].agent == "v11_glm5"


def test_chat_conversation_has_no_parent(telegram_interface):
    """Only topics inherit — a plain chat must not fall back to anything."""
    assert telegram_interface.parent_conversation_id("tg--100") is None
    assert telegram_interface.parent_conversation_id("tg-123") is None
    assert telegram_interface.parent_conversation_id("tg--100-topic-7") == "tg--100"
