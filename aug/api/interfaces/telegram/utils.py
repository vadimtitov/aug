"""Shared helpers for Telegram bot handlers."""

from collections.abc import Callable
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from aug.config import get_settings
from aug.utils.state import TelegramChatState, load_state


def is_allowed(user_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or user_id in allowed


def restricted(handler: Callable) -> Callable:
    """Decorator: silently drop updates from users not on the allow-list.

    Apply to every command/callback handler so auth is enforced at the boundary
    and cannot be accidentally omitted from a new handler.
    """

    @wraps(handler)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user and is_allowed(update.effective_user.id):
            return await handler(self, update, context)

    return wrapper


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def get_thread_id(chat_id: int, topic_id: int | None = None) -> str:
    if topic_id is not None:
        return f"tg-{chat_id}-topic-{topic_id}"
    session = load_state().telegram.chats.get(str(chat_id), TelegramChatState()).session
    return f"tg-{chat_id}-{session}"
