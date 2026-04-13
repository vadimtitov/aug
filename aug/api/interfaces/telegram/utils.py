"""Shared helpers for Telegram bot handlers."""

from collections.abc import Callable
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from aug.config import get_settings
from aug.utils.state import get_state


def _is_allowed(chat_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or chat_id in allowed


def _restricted(handler: Callable) -> Callable:
    """Decorator: silently drop updates from chats not on the allow-list.

    Apply to every command/callback handler so auth is enforced at the boundary
    and cannot be accidentally omitted from a new handler.
    """

    @wraps(handler)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat and _is_allowed(update.effective_chat.id):
            return await handler(self, update, context)

    return wrapper


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _thread_id(chat_id: int) -> str:
    session = get_state("telegram", "chats", str(chat_id), "session", default=0)
    return f"tg-{chat_id}-{session}"
