"""Shared helpers for Telegram bot handlers."""

import re
from collections.abc import Callable
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from aug.config import get_settings
from aug.utils.state import TelegramChatState, load_state

# Matches a non-topic thread ID: tg-{chat_id}-{session}.
_CHAT_THREAD_RE = re.compile(r"^tg-(-?\d+)-(\d+)$")


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


def get_thread_id(chat_id: int, topic_id: int | None) -> str:
    if topic_id is not None:
        return f"tg-{chat_id}-topic-{topic_id}"
    session = load_state().telegram.chats.get(str(chat_id), TelegramChatState()).session
    return f"tg-{chat_id}-{session}"


def get_conversation_id(thread_id: str) -> str:
    """Return the conversation a Telegram *thread_id* belongs to.

    A forum topic is its own conversation, so its thread ID is already the key.
    A plain chat gets a fresh thread ID on every /clear (the trailing session
    counter), so the counter is stripped to keep per-chat settings across resets.
    """
    m = _CHAT_THREAD_RE.match(thread_id)
    return f"tg-{m.group(1)}" if m else thread_id
