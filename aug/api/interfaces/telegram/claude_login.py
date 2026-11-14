"""Telegram handlers for the Claude login (/claude_login command).

Provides _ClaudeLoginMixin, which TelegramInterface inherits.

Flow:
  /claude_login → login status + a sign-in button →
  the user signs in and pastes the code the page shows →
  the code message is deleted at once (it never reaches the agent) →
  the token is stored in hushed → "Claude is connected until …"

A sign-in waits for its code from the user who started it, in the chat and topic it
was started in; nothing else is held back.  Any other text there ends it.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import CommandHandler, ContextTypes, MessageHandler, filters

from aug.api.interfaces.telegram.utils import restricted
from aug.core.harness.claude_code.login import LoginAttempt, login_status, start_login
from aug.utils.claude_setup_token import SetupTokenError, looks_like_code
from aug.utils.hushed import HushedError

logger = logging.getLogger(__name__)

_LOGIN_TIMEOUT_SECONDS = 600

_PROMPT = (
    "{status}\n\n"
    "1. Tap the button and sign in to Claude.\n"
    "2. Copy the code the page shows and paste it here.\n\n"
    "🔒 The code is not sent to the AI: AUG stores the login in its secrets and "
    "deletes your message straight away.\n"
    "/cancel to stop."
)
_NOT_A_CODE = (
    "That isn't the sign-in code, so it was deleted and not used, and the sign-in "
    "was stopped. Send /claude_login to start again."
)
_REFUSED = "Claude did not accept that code. Send /claude_login to try again."
_FAILED = "The Claude sign-in failed. Send /claude_login to try again."
_TIMED_OUT = "The Claude sign-in timed out. Send /claude_login to start again."

# Who a sign-in waits on: chat, forum topic (None outside topics), user.
_Key = tuple[int, int | None, int]


@dataclass
class _Pending:
    attempt: LoginAttempt
    prompt: Message
    timeout: asyncio.Task | None = None


class _PendingLogins:
    """The sign-ins waiting for their code, one per chat, topic and user."""

    def __init__(self) -> None:
        self._by_key: dict[_Key, _Pending] = {}

    def __contains__(self, key: _Key) -> bool:
        return key in self._by_key

    async def add(self, key: _Key, attempt: LoginAttempt, prompt: Message) -> None:
        """Wait on *key* for the code; a sign-in already waiting there is ended."""
        await self.end(key)
        pending = _Pending(attempt, prompt)
        pending.timeout = asyncio.create_task(self._time_out(key, pending))
        self._by_key[key] = pending

    def take(self, key: _Key) -> _Pending | None:
        """Stop waiting on *key*; the caller finishes the sign-in."""
        pending = self._by_key.pop(key, None)
        if pending is not None and pending.timeout is not None:
            pending.timeout.cancel()
        return pending

    async def end(self, key: _Key) -> bool:
        """End the sign-in waiting on *key*, if any; True when there was one."""
        pending = self.take(key)
        if pending is None:
            return False
        await _close(pending)
        return True

    async def end_all(self) -> None:
        for key in list(self._by_key):
            await self.end(key)

    async def _time_out(self, key: _Key, pending: _Pending) -> None:
        await asyncio.sleep(_LOGIN_TIMEOUT_SECONDS)
        if self._by_key.get(key) is pending:
            del self._by_key[key]
            await _close(pending)
            with contextlib.suppress(Exception):
                await pending.prompt.chat.send_message(
                    _TIMED_OUT, message_thread_id=_thread(pending.prompt)
                )


class _AwaitingCode(filters.MessageFilter):
    """Messages from a user whose sign-in is waiting in that chat and topic."""

    def __init__(self, logins: _PendingLogins) -> None:
        super().__init__(name="awaiting_claude_login_code")
        self._logins = logins

    def filter(self, message: Message) -> bool:
        return message.from_user is not None and _key(message) in self._logins


class _ClaudeLoginMixin:
    """Mixin that adds /claude_login to TelegramInterface."""

    def build_claude_login_handlers(self, bot_app) -> None:
        """Register the /claude_login handlers on *bot_app*, before the agent's."""
        self._claude_logins = _PendingLogins()
        awaiting = _AwaitingCode(self._claude_logins)
        bot_app.add_handler(CommandHandler("claude_login", self._claude_login_start))
        bot_app.add_handler(CommandHandler("cancel", self._claude_login_cancel, filters=awaiting))
        bot_app.add_handler(
            MessageHandler(
                filters.UpdateType.MESSAGE & filters.TEXT & ~filters.COMMAND & awaiting,
                self._claude_login_code,
            )
        )

    async def end_claude_logins(self) -> None:
        """End every waiting sign-in (shutdown)."""
        await self._claude_logins.end_all()

    @restricted
    async def _claude_login_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None or message.from_user is None:
            return
        try:
            attempt = await start_login()
        except SetupTokenError as e:
            logger.error("claude login could not start: %s", e)
            await message.reply_text("Could not start the Claude sign-in. See AUG's logs.")
            return
        try:
            prompt = await message.reply_text(
                _PROMPT.format(status=_status_line()),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Sign in to Claude", url=attempt.url)]]
                ),
            )
        except BaseException:
            await attempt.cancel()
            raise
        await self._claude_logins.add(_key(message), attempt, prompt)

    @restricted
    async def _claude_login_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return
        pending = self._claude_logins.take(_key(message))
        code = message.text or ""
        with contextlib.suppress(Exception):
            await message.delete()
        if pending is None:
            return
        thread = _thread(message)
        if not looks_like_code(code):
            await _close(pending)
            await message.chat.send_message(_NOT_A_CODE, message_thread_id=thread)
            return
        with contextlib.suppress(Exception):
            await pending.prompt.delete()  # its link was for this one sign-in
        try:
            status = await pending.attempt.complete(code)
        except SetupTokenError as e:
            logger.warning("claude login failed: %s", e)
            await message.chat.send_message(_REFUSED, message_thread_id=thread)
            return
        except HushedError as e:
            logger.error("claude login token could not be stored: %s", e)
            await message.chat.send_message(
                "Signed in, but the login could not be saved. See AUG's logs.",
                message_thread_id=thread,
            )
            return
        except Exception as e:
            logger.error("claude login failed: %s", type(e).__name__)
            await message.chat.send_message(_FAILED, message_thread_id=thread)
            return
        assert status.expires_at is not None
        await message.chat.send_message(
            f"✅ Claude is connected until {status.expires_at:%d.%m.%Y}. "
            "Chats pick up the new login with their next message.",
            message_thread_id=thread,
        )

    @restricted
    async def _claude_login_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.message
        if message is not None and await self._claude_logins.end(_key(message)):
            await message.reply_text("Sign-in cancelled.")


def _key(message: Message) -> _Key:
    user = message.from_user.id if message.from_user else 0
    return (message.chat.id, _thread(message), user)


def _thread(message: Message) -> int | None:
    return message.message_thread_id if message.is_topic_message else None


def _status_line() -> str:
    status = login_status()
    if status.expires_at is None:
        return "Sign in to Claude for AUG's Claude agents. The login lasts a year."
    return (
        f"The current Claude login is valid until {status.expires_at:%d.%m.%Y}. "
        "Sign in again to renew it for a year."
    )


async def _close(pending: _Pending) -> None:
    """End a sign-in that won't complete: stop the CLI, remove the link."""
    await pending.attempt.cancel()
    with contextlib.suppress(Exception):
        await pending.prompt.delete()
