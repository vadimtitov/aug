"""The Codex login: a ChatGPT sign-in by device code, from /codex_login in Telegram.

Codex's side is scripted (``FakeLoginRuntime``); the real app-server's login is not
exercised here, since it needs a ChatGPT account.
"""

import asyncio
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.telegram.interface import TelegramInterface
from aug.core.harness.codex.login import (
    CodexAccount,
    LoginFailed,
    login_status,
    start_login,
)

ACCOUNT = {"type": "chatgpt", "email": "me@example.com", "planType": "plus"}


class FakeLoginRuntime:
    """Answers the login requests; ``finish()`` ends the sign-in as Codex would."""

    def __init__(self) -> None:
        self.account: dict | None = None
        self.requests: list[str] = []
        self.waits: dict[str, asyncio.Future[str | None]] = {}

    async def request(self, method: str, params: dict | None = None) -> dict:
        self.requests.append(method)
        if method == "account/login/start":
            return {
                "loginId": "L1",
                "verificationUrl": "https://auth.openai.com/codex/device",
                "userCode": "AB-12",
            }
        if method == "account/read":
            return {"account": self.account}
        return {}

    def login_finished(self, login_id: str) -> asyncio.Future[str | None]:
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self.waits[login_id] = future
        return future

    def finish(self, error: str | None = None) -> None:
        if error is None:
            self.account = ACCOUNT
        self.waits["L1"].set_result(error)


@pytest.fixture
def runtime(monkeypatch) -> FakeLoginRuntime:
    """The runtime the Telegram flow signs in with."""
    runtime = FakeLoginRuntime()
    flow = "aug.api.interfaces.telegram.codex_login"
    monkeypatch.setattr(f"{flow}.login_status", partial(login_status, runtime))
    monkeypatch.setattr(f"{flow}.start_login", partial(start_login, runtime))
    return runtime


async def test_a_sign_in_ends_with_the_account(runtime):
    attempt = await start_login(runtime)  # type: ignore[arg-type]
    assert (attempt.url, attempt.code) == ("https://auth.openai.com/codex/device", "AB-12")
    runtime.finish()
    assert await attempt.wait() == CodexAccount(email="me@example.com", plan="plus")


async def test_a_refused_sign_in_fails(runtime):
    attempt = await start_login(runtime)  # type: ignore[arg-type]
    runtime.finish("access denied")
    with pytest.raises(LoginFailed, match="access denied"):
        await attempt.wait()


async def test_a_cancelled_sign_in_tells_codex_and_ends_the_wait(runtime):
    attempt = await start_login(runtime)  # type: ignore[arg-type]
    waiting = asyncio.create_task(attempt.wait())
    await attempt.cancel()
    with pytest.raises(LoginFailed, match="cancelled"):
        await waiting
    assert "account/login/cancel" in runtime.requests


async def test_a_code_left_unused_expires(runtime, monkeypatch):
    monkeypatch.setattr("aug.core.harness.codex.login.LOGIN_TIMEOUT_SECONDS", 0.01)
    attempt = await start_login(runtime)  # type: ignore[arg-type]
    with pytest.raises(LoginFailed, match="expired"):
        await attempt.wait()
    assert "account/login/cancel" in runtime.requests


# -- Telegram ----------------------------------------------------------------------------


@pytest.fixture
def bot():
    with patch("aug.api.interfaces.telegram.utils.is_allowed", return_value=True):
        interface = TelegramInterface()
        interface.build_claude_login_handlers(MagicMock())
        interface.build_codex_login_handlers(MagicMock())
        yield interface


def _message(text: str, *, topic: int | None = None) -> MagicMock:
    message = MagicMock()
    message.text = text
    message.chat.id = 1
    message.chat.type = "private"
    message.chat.send_message = AsyncMock()
    message.from_user.id = 1
    message.is_topic_message = topic is not None
    message.message_thread_id = topic
    prompt = MagicMock()
    prompt.edit_text = AsyncMock()
    prompt.chat = message.chat
    message.reply_text = AsyncMock(return_value=prompt)
    return message


def _update(message: MagicMock) -> MagicMock:
    update = MagicMock()
    update.message = message
    update.effective_user.id = message.from_user.id
    return update


async def _pending_task(bot: Any) -> asyncio.Task:
    (pending,) = bot._codex_logins.all()
    assert pending.task is not None
    return pending.task


async def test_telegram_sign_in_shows_the_code_and_says_when_it_is_done(runtime, bot):
    start = _message("/codex_login", topic=7)
    await bot._codex_login_start(_update(start), MagicMock())
    (text,), kwargs = start.reply_text.call_args
    assert "<code>AB-12</code>" in text and "Nothing to paste back" in text
    (button,) = kwargs["reply_markup"].inline_keyboard[0]
    assert button.url == "https://auth.openai.com/codex/device"
    assert bot.codex_login_waiting.filter(_message("/cancel", topic=7))
    assert not bot.codex_login_waiting.filter(_message("/cancel", topic=8))

    task = await _pending_task(bot)
    runtime.finish()
    await task
    (done,), _ = start.reply_text.return_value.edit_text.call_args
    assert "Codex is signed in as me@example.com (plus)" in done
    assert not bot._codex_logins.all()


async def test_telegram_cancel_ends_the_sign_in(runtime, bot):
    start = _message("/codex_login")
    await bot._codex_login_start(_update(start), MagicMock())
    await bot._cancel_sign_ins(_update(_message("/cancel")), MagicMock())
    assert "account/login/cancel" in runtime.requests
    (text,), _ = start.reply_text.return_value.edit_text.call_args
    assert "cancelled" in text
    assert not bot._codex_logins.all()


async def test_a_new_sign_in_replaces_the_one_waiting(runtime, bot):
    first = _message("/codex_login")
    await bot._codex_login_start(_update(first), MagicMock())
    await bot._codex_login_start(_update(_message("/codex_login")), MagicMock())
    (text,), _ = first.reply_text.return_value.edit_text.call_args
    assert "Replaced" in text
    assert len(bot._codex_logins.all()) == 1
    await bot.end_codex_logins()


async def test_a_sign_in_that_worked_counts_even_if_the_account_is_unreadable(runtime):
    attempt = await start_login(runtime)  # type: ignore[arg-type]
    runtime.finish()

    async def unreadable(method: str, params: dict | None = None) -> dict:
        raise TimeoutError

    runtime.request = unreadable  # type: ignore[method-assign]
    assert await attempt.wait() == CodexAccount(email="", plan="")


async def test_the_sign_in_is_only_started_in_a_private_chat(runtime, bot):
    start = _message("/codex_login")
    start.chat.type = "supergroup"
    await bot._codex_login_start(_update(start), MagicMock())
    (text,), _ = start.reply_text.call_args
    assert "private chat" in text
    assert "account/login/start" not in runtime.requests


async def test_a_sign_in_codex_cannot_start_says_so(bot, monkeypatch):
    async def broken() -> Any:
        raise RuntimeError("codex binary missing")

    monkeypatch.setattr("aug.api.interfaces.telegram.codex_login.login_status", broken)
    start = _message("/codex_login")
    await bot._codex_login_start(_update(start), MagicMock())
    (text,), _ = start.reply_text.call_args
    assert "Could not start" in text
    assert not bot._codex_logins.all()
