"""The allow-list: empty means nobody may use the bot; a message then gets its IDs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.ext import ApplicationHandlerStop

from aug.api.interfaces.telegram.interface import TelegramInterface
from aug.api.interfaces.telegram.utils import is_allowed


def _settings(allowed: set[int]) -> MagicMock:
    return MagicMock(allowed_chat_ids=allowed)


def _update(user_id: int, chat_id: int) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.effective_message.reply_text = AsyncMock()
    return update


@pytest.mark.parametrize(
    ("allowed", "user_id", "expected"),
    [(set(), 1, False), ({1}, 1, True), ({1}, 2, False)],
)
def test_is_allowed(allowed, user_id, expected):
    with patch("aug.api.interfaces.telegram.utils.get_settings", return_value=_settings(allowed)):
        assert is_allowed(user_id) is expected


@pytest.mark.asyncio
async def test_without_an_allow_list_a_message_gets_only_its_ids():
    update = _update(user_id=42, chat_id=42)
    with patch("aug.api.interfaces.telegram.interface.get_settings", return_value=_settings(set())):
        with pytest.raises(ApplicationHandlerStop):
            await TelegramInterface()._handle_unconfigured(update, MagicMock())
    text = update.effective_message.reply_text.await_args.args[0]
    assert "Your Telegram ID: 42." in text
    assert "TELEGRAM_ALLOWED_CHAT_IDS" in text


@pytest.mark.asyncio
async def test_in_a_group_the_reply_names_the_chat_too():
    update = _update(user_id=42, chat_id=-100)
    with patch("aug.api.interfaces.telegram.interface.get_settings", return_value=_settings(set())):
        with pytest.raises(ApplicationHandlerStop):
            await TelegramInterface()._handle_unconfigured(update, MagicMock())
    assert "42 (this chat: -100)" in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_with_an_allow_list_messages_pass_through_silently():
    update = _update(user_id=7, chat_id=7)
    with patch("aug.api.interfaces.telegram.interface.get_settings", return_value=_settings({1})):
        await TelegramInterface()._handle_unconfigured(update, MagicMock())
    update.effective_message.reply_text.assert_not_awaited()
