"""Shared helpers for Telegram bot handlers."""

from collections.abc import Callable
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from aug.config import get_settings
from aug.utils.state import TelegramChatState, load_state


def is_allowed(chat_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or chat_id in allowed


def restricted(handler: Callable) -> Callable:
    """Decorator: silently drop updates from chats not on the allow-list.

    Apply to every command/callback handler so auth is enforced at the boundary
    and cannot be accidentally omitted from a new handler.
    """

    @wraps(handler)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat and is_allowed(update.effective_chat.id):
            return await handler(self, update, context)

    return wrapper


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def get_thread_id(chat_id: int) -> str:
    session = load_state().telegram.chats.get(str(chat_id), TelegramChatState()).session
    return f"tg-{chat_id}-{session}"
