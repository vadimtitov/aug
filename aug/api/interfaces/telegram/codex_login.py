"""Telegram handlers for the Codex login (/codex_login command).

Provides _CodexLoginMixin, which TelegramInterface inherits.

Flow:
  /codex_login → login status, a sign-in button and a short code →
  the user signs in to ChatGPT on the page and enters the code there →
  the message turns into "Codex is signed in as …"

Nothing is typed back into the chat, so nothing can reach the agent.  One sign-in waits
per chat, topic and user; a new /codex_login there replaces it, /cancel (the
interface's, through ``codex_login_waiting``) ends it.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from html import escape

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import CommandHandler, ContextTypes, filters

from aug.api.interfaces.telegram.utils import restricted
from aug.core.harness.codex.login import (
    CodexAccount,
    LoginAttempt,
    LoginFailed,
    login_status,
    start_login,
)

logger = logging.getLogger(__name__)

_PROMPT = (
    "{status}\n\n"
    "1. Tap the button and sign in with your ChatGPT account.\n"
    "2. Enter this code there:\n\n"
    "<code>{code}</code>\n\n"
    "Nothing to paste back here: this message updates once you are signed in.\n"
    "/cancel to stop."
)
_COULD_NOT_START = "Could not start the Codex sign-in. See AUG's logs."
# Its code signs AUG in to whoever enters it first, and its status names the account.
_PRIVATE_ONLY = "Send /codex_login in a private chat with me: the sign-in is yours alone."
_FAILED = "The Codex sign-in did not complete ({reason}). Send /codex_login to try again."
_CANCELLED = "Codex sign-in cancelled."
_REPLACED = "Replaced by a newer /codex_login."
_SIGNED_IN = "✅ Codex is signed in as {who}. Chats use the login from their next message."

# Who a sign-in waits on: chat, forum topic (None outside topics), user.
_Key = tuple[int, int | None, int]


@dataclass
class _Pending:
    attempt: LoginAttempt
    prompt: Message
    task: asyncio.Task[None] | None = None


class _PendingLogins:
    """The sign-ins in progress, one per chat, topic and user."""

    def __init__(self) -> None:
        self._by_key: dict[_Key, _Pending] = {}

    def __contains__(self, key: _Key) -> bool:
        return key in self._by_key

    def add(self, key: _Key, pending: _Pending) -> _Pending | None:
        """Track *pending* on *key*; returns the sign-in it replaces, if any."""
        old = self._by_key.get(key)
        self._by_key[key] = pending
        return old

    def take(self, key: _Key, pending: _Pending | None = None) -> _Pending | None:
        """Stop tracking *key* (only if it is still *pending*, when given)."""
        current = self._by_key.get(key)
        if current is None or (pending is not None and current is not pending):
            return None
        return self._by_key.pop(key)

    def all(self) -> list[_Pending]:
        return list(self._by_key.values())


class _SigningIn(filters.MessageFilter):
    """Messages from a user whose Codex sign-in is in progress in that chat and topic."""

    def __init__(self, logins: _PendingLogins) -> None:
        super().__init__(name="codex_login_in_progress")
        self._logins = logins

    def filter(self, message: Message) -> bool:
        return message.from_user is not None and _key(message) in self._logins


class _CodexLoginMixin:
    """Mixin that adds /codex_login to TelegramInterface."""

    def build_codex_login_handlers(self, bot_app) -> None:
        """Register /codex_login on *bot_app*, before the agent's handlers."""
        self._codex_logins = _PendingLogins()
        self.codex_login_waiting = _SigningIn(self._codex_logins)
        bot_app.add_handler(CommandHandler("codex_login", self._codex_login_start))

    async def cancel_codex_login(self, message: Message) -> None:
        """End the sign-in in progress in *message*'s chat and topic for its user."""
        pending = self._codex_logins.take(_key(message))
        if pending is not None:
            await _close(pending, _CANCELLED)

    async def end_codex_logins(self) -> None:
        """End every sign-in in progress (shutdown)."""
        for pending in self._codex_logins.all():
            if pending.task is not None:
                pending.task.cancel()
            with contextlib.suppress(Exception):
                await pending.attempt.cancel()

    @restricted
    async def _codex_login_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None or message.from_user is None:
            return
        if message.chat.type != ChatType.PRIVATE:
            await message.reply_text(_PRIVATE_ONLY)
            return
        try:
            account = await login_status()
            attempt = await start_login()
        except Exception:
            logger.exception("codex login could not start")
            await message.reply_text(_COULD_NOT_START)
            return
        try:
            prompt = await message.reply_text(
                _PROMPT.format(status=escape(_status_line(account)), code=escape(attempt.code)),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Sign in to ChatGPT", url=attempt.url)]]
                ),
            )
        except BaseException:
            await attempt.cancel()
            raise
        key = _key(message)
        pending = _Pending(attempt, prompt)
        old = self._codex_logins.add(key, pending)
        pending.task = asyncio.create_task(self._codex_login_finish(key, pending))
        if old is not None:
            await _close(old, _REPLACED)

    async def _codex_login_finish(self, key: _Key, pending: _Pending) -> None:
        """Wait for the sign-in and say how it went, in place of the sign-in message."""
        try:
            account = await pending.attempt.wait()
        except LoginFailed as e:
            logger.warning("codex login failed: %s", e)
            if self._codex_logins.take(key, pending) is not None:
                await _edit(pending.prompt, _FAILED.format(reason=e))
            return
        except Exception:
            logger.exception("codex login failed")
            if self._codex_logins.take(key, pending) is not None:
                await _edit(pending.prompt, _FAILED.format(reason="an error, see AUG's logs"))
            return
        self._codex_logins.take(key, pending)
        await _edit(pending.prompt, _SIGNED_IN.format(who=_who(account)))


def _key(message: Message) -> _Key:
    user = message.from_user.id if message.from_user else 0
    return (message.chat.id, _thread(message), user)


def _thread(message: Message) -> int | None:
    return message.message_thread_id if message.is_topic_message else None


def _who(account: CodexAccount) -> str:
    name = account.email or "your ChatGPT account"
    return f"{name} ({account.plan})" if account.plan else name


def _status_line(account: CodexAccount | None) -> str:
    if account is None:
        return "Codex is not signed in. Sign in with ChatGPT for AUG's Codex agents."
    return f"Codex is signed in as {_who(account)}. Sign in again to replace the login."


async def _close(pending: _Pending, text: str) -> None:
    """End a sign-in that won't complete: stop it, say why in place of its message."""
    if pending.task is not None:
        pending.task.cancel()
    with contextlib.suppress(Exception):
        await pending.attempt.cancel()
    await _edit(pending.prompt, text)


async def _edit(prompt: Message, text: str) -> None:
    """Replace the sign-in message (and its button, whose link is spent)."""
    try:
        await prompt.edit_text(text, reply_markup=None)
    except Exception:
        logger.warning("codex login message could not be edited", exc_info=True)
        with contextlib.suppress(Exception):
            await prompt.chat.send_message(text, message_thread_id=_thread(prompt))
