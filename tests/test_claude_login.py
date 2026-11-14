"""The Claude login: /claude_login drives ``claude setup-token`` (a stub here), keeps the
code away from the agent, stores the token in hushed only."""

import asyncio
import sys
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.telegram.claude_login import _AwaitingCode
from aug.api.interfaces.telegram.interface import TelegramInterface
from aug.core.harness import HarnessError
from aug.core.harness.base import error_text
from aug.core.harness.claude_code import harness as cc
from aug.core.harness.claude_code import login as claude_login
from aug.core.harness.claude_code.login import login_status, start_login
from aug.core.harness.claude_code.session import _result_error
from aug.utils.claude_setup_token import SetupToken, SetupTokenError, looks_like_code
from aug.utils.state import update_state

TOKEN = "sk-ant-oat01-" + "T" * 90
CODE = "good-code#the-state"

# Behaves like `claude setup-token` on a terminal: escapes, a link, a prompt, a token.
_STUB = f"""#!{sys.executable}
import sys
w = sys.stdout.write
w("\\x1b[?25l\\x1b[1mWelcome to Claude Code\\x1b[22m\\r\\n")
w("Browser didn't open? Use the url below to sign in\\r\\n\\r\\n")
w("https://claude.com/cai/oauth/authorize?code=true&client_id=abc&state=xyz\\r\\n\\r\\n")
w("Paste code here if prompted > ")
sys.stdout.flush()
code = sys.stdin.readline().strip()
if code == {CODE!r}:
    w("\\r\\nYour OAuth token (valid for 1 year):\\r\\n\\r\\n\\x1b[1m{TOKEN}\\x1b[22m\\r\\n\\r\\n")
    w("Store this token securely.\\r\\n")
    sys.stdout.flush()
elif code.startswith("hang"):
    import time
    time.sleep(60)
else:
    # Like the real CLI: it says so, then waits for Enter to retry.
    w("\\r\\nOAuth error: That code wasn't accepted.\\r\\nPress Enter to retry.\\r\\n")
    sys.stdout.flush()
    sys.stdin.readline()
"""


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("aug.utils.data.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def stub_cli(tmp_path, monkeypatch):
    cli = tmp_path / "claude"
    cli.write_text(_STUB)
    cli.chmod(0o755)
    monkeypatch.setattr(claude_login, "BUNDLED_CLI", cli)
    return cli


@pytest.fixture
def hushed(monkeypatch):
    stored: dict[str, str] = {}
    monkeypatch.setattr(
        claude_login, "write_secret", lambda name, value: stored.update({name: value})
    )
    return stored


# -- the CLI driver --------------------------------------------------------------------


async def test_the_link_and_the_token_are_read_from_the_terminal(stub_cli):
    setup = SetupToken(stub_cli)
    try:
        url = await setup.start()
        assert url == "https://claude.com/cai/oauth/authorize?code=true&client_id=abc&state=xyz"
        assert await setup.finish(CODE) == TOKEN
    finally:
        await setup.close()


async def test_a_refused_code_is_an_error_at_once(stub_cli):
    setup = SetupToken(stub_cli)
    try:
        await setup.start()
        started = time.monotonic()
        with pytest.raises(SetupTokenError, match="wasn't accepted"):
            await setup.finish("bad#code")
        assert time.monotonic() - started < 5
    finally:
        await setup.close()


async def test_a_cli_that_prints_no_link_is_an_error(tmp_path, monkeypatch):
    cli = tmp_path / "silent"
    cli.write_text(f"#!{sys.executable}\nprint('nothing to see')\n")
    cli.chmod(0o755)
    setup = SetupToken(cli)
    with pytest.raises(SetupTokenError, match="no sign-in link"):
        await setup.start()
    await setup.close()


async def test_a_cli_that_never_answers_leaves_the_code_out_of_the_error(stub_cli, monkeypatch):
    monkeypatch.setattr("aug.utils.claude_setup_token._TOKEN_TIMEOUT", 1.0)
    setup = SetupToken(stub_cli)
    try:
        await setup.start()
        with pytest.raises(SetupTokenError) as raised:
            await setup.finish("hang-secret-code#state")
        assert "secret-code" not in str(raised.value)
    finally:
        await setup.close()


def test_only_the_sign_in_code_shape_is_taken_as_a_code():
    assert looks_like_code("el5lzaKb5Asy0dUp#-Ac40JOX_8X0xCz")
    assert not looks_like_code("what's the weather?")
    assert not looks_like_code("abc")


# -- the login -------------------------------------------------------------------------


async def test_a_login_stores_the_token_in_hushed_and_dates_it(stub_cli, hushed):
    attempt = await start_login()
    status = await attempt.complete(CODE)
    assert hushed == {cc.TOKEN_SECRET: TOKEN}
    assert status.expires_at is not None
    assert login_status().expires_at == status.expires_at


async def test_a_refused_code_stores_nothing(stub_cli, hushed):
    attempt = await start_login()
    with pytest.raises(SetupTokenError):
        await attempt.complete("bad#code")
    assert hushed == {}
    assert login_status().issued_at is None


async def test_the_token_is_never_logged(stub_cli, hushed, caplog):
    caplog.set_level("DEBUG")
    attempt = await start_login()
    await attempt.complete(CODE)
    assert TOKEN not in caplog.text


async def test_a_new_login_restarts_claude_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "WORKSPACE_DIR", tmp_path / "ws")
    harness = cc.ClaudeCodeHarness(
        cc.ClaudeCodeProfile(model="m", effort="high", builtin_tools=(), tools=[]),
        config_dir=tmp_path / "claude",
    )
    record = cc._Record(context_id="c", session_id="s", interface="rest_api", prompt="p", skills=[])
    before = harness._fingerprint(record)
    async with update_state() as state:
        state.claude_login.issued_at = datetime.now(UTC).isoformat()
    assert harness._fingerprint(record) != before


# -- errors ----------------------------------------------------------------------------


def test_a_rejected_claude_login_asks_for_a_new_one():
    result = MagicMock(subtype="error_during_execution", api_error_status=401, errors=[])
    assert _result_error(result).kind == "login"
    assert "/claude_login" in error_text(HarnessError(kind="login"))


# -- Telegram --------------------------------------------------------------------------


@pytest.fixture
def bot():
    with patch("aug.api.interfaces.telegram.utils.is_allowed", return_value=True):
        interface = TelegramInterface()
        interface.build_claude_login_handlers(MagicMock())
        yield interface


def _message(text: str, *, topic: int | None = None, user: int = 1) -> MagicMock:
    message = MagicMock()
    message.text = text
    message.chat.id = -100
    message.chat.send_message = AsyncMock()
    message.from_user.id = user
    message.is_topic_message = topic is not None
    message.message_thread_id = topic
    message.delete = AsyncMock()
    prompt = MagicMock()
    prompt.delete = AsyncMock()
    prompt.chat = message.chat
    prompt.is_topic_message = topic is not None
    prompt.message_thread_id = topic
    message.reply_text = AsyncMock(return_value=prompt)
    return message


def _update(message: MagicMock) -> MagicMock:
    update = MagicMock()
    update.message = message
    update.effective_user.id = message.from_user.id
    return update


def _awaiting(bot: TelegramInterface, message: MagicMock) -> bool:
    return _AwaitingCode(bot._claude_logins).filter(message)


async def test_telegram_login_end_to_end(stub_cli, hushed, bot):
    start = _message("/claude_login", topic=7)
    await bot._claude_login_start(_update(start), MagicMock())
    (prompt,), kwargs = start.reply_text.call_args
    assert "not sent to the AI" in prompt
    (button,) = kwargs["reply_markup"].inline_keyboard[0]
    assert button.url.startswith("https://claude.com/cai/oauth/authorize")

    code = _message(CODE, topic=7)
    assert _awaiting(bot, code)
    await bot._claude_login_code(_update(code), MagicMock())
    code.delete.assert_awaited()
    assert hushed == {cc.TOKEN_SECRET: TOKEN}
    (reply,), kwargs = code.chat.send_message.call_args
    assert "Claude is connected until" in reply and TOKEN not in reply
    assert kwargs["message_thread_id"] == 7
    start.reply_text.return_value.delete.assert_awaited()  # the one-time link is gone
    assert not _awaiting(bot, _message("hello", topic=7))


async def test_telegram_only_the_starting_topic_and_user_are_held(stub_cli, bot):
    await bot._claude_login_start(_update(_message("/claude_login", topic=7)), MagicMock())
    assert _awaiting(bot, _message("x", topic=7))
    assert not _awaiting(bot, _message("x", topic=8))
    assert not _awaiting(bot, _message("x"))
    assert not _awaiting(bot, _message("x", topic=7, user=2))
    await bot.end_claude_logins()


async def test_telegram_text_that_is_no_code_is_deleted_and_ends_the_sign_in(stub_cli, hushed, bot):
    await bot._claude_login_start(_update(_message("/claude_login")), MagicMock())
    chatter = _message("hey, what's up?")
    await bot._claude_login_code(_update(chatter), MagicMock())
    chatter.delete.assert_awaited()
    assert hushed == {}
    assert not _awaiting(bot, _message("next message"))


async def test_telegram_a_second_start_ends_the_first(stub_cli, bot):
    first = _message("/claude_login")
    await bot._claude_login_start(_update(first), MagicMock())
    (pending,) = bot._claude_logins._by_key.values()
    await bot._claude_login_start(_update(_message("/claude_login")), MagicMock())
    assert pending.attempt._setup._process.poll() is not None
    first.reply_text.return_value.delete.assert_awaited()
    assert len(bot._claude_logins._by_key) == 1
    await bot.end_claude_logins()


async def test_telegram_cancel_ends_the_cli(stub_cli, bot):
    await bot._claude_login_start(_update(_message("/claude_login")), MagicMock())
    (pending,) = bot._claude_logins._by_key.values()
    cancel = _message("/cancel")
    await bot._claude_login_cancel(_update(cancel), MagicMock())
    assert pending.attempt._setup._process.poll() is not None
    cancel.reply_text.assert_awaited_with("Sign-in cancelled.")


async def test_telegram_a_forgotten_sign_in_times_out(stub_cli, bot, monkeypatch):
    monkeypatch.setattr("aug.api.interfaces.telegram.claude_login._LOGIN_TIMEOUT_SECONDS", 0.1)
    start = _message("/claude_login")
    await bot._claude_login_start(_update(start), MagicMock())
    (pending,) = bot._claude_logins._by_key.values()
    await asyncio.sleep(0.5)
    assert not bot._claude_logins._by_key
    assert pending.attempt._setup._process.poll() is not None
    (text,), _ = start.chat.send_message.call_args
    assert "timed out" in text


def test_telegram_the_code_is_caught_before_the_agent_sees_text():
    with patch("aug.api.interfaces.telegram.interface.get_settings") as settings:
        settings.return_value.TELEGRAM_BOT_TOKEN = "1:x"
        settings.return_value.TELEGRAM_BASE_URL = None
        handlers = TelegramInterface().build_bot().handlers[0]
    names = [getattr(getattr(h, "callback", None), "__name__", "") for h in handlers]
    assert names.index("_claude_login_code") < names.index("_handle_text")
