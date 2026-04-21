"""Tests for _TelegramRateLimiter — token bucket for Telegram API calls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.telegram.interface import _spinner_task, _TelegramRateLimiter
from tests.test_send_stream import _make_context, _stream


@pytest.fixture()
def interface():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    return TelegramInterface(checkpointer=MagicMock())


@pytest.mark.asyncio
async def test_throttle_returns_immediately_when_tokens_available():
    limiter = _TelegramRateLimiter(rate=2.0, capacity=5.0)
    slept = []

    sleep_mock = AsyncMock(side_effect=slept.append)
    with patch("aug.api.interfaces.telegram.interface.asyncio.sleep", sleep_mock):
        await limiter.throttle()

    assert slept == []


def test_try_acquire_returns_true_and_consumes_token():
    limiter = _TelegramRateLimiter(rate=2.0, capacity=3.0)
    assert limiter.try_acquire() is True
    assert limiter._tokens == 2.0


def test_try_acquire_returns_false_when_exhausted():
    limiter = _TelegramRateLimiter(rate=2.0, capacity=2.0)
    limiter.try_acquire()
    limiter.try_acquire()
    assert limiter.try_acquire() is False
    # tokens must not go negative
    assert limiter._tokens >= 0.0


@pytest.mark.asyncio
async def test_throttle_waits_when_exhausted_then_completes():
    """When tokens are empty, throttle() sleeps for the refill duration, then returns."""
    fake_time = [0.0]

    _iface = "aug.api.interfaces.telegram.interface"
    mono_patch = patch(f"{_iface}._monotonic", side_effect=lambda: fake_time[0])
    with mono_patch:
        limiter = _TelegramRateLimiter(rate=2.0, capacity=1.0)
        limiter.try_acquire()  # drain the single token

        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)
            fake_time[0] += seconds  # advance time so refill adds tokens

        with patch("aug.api.interfaces.telegram.interface.asyncio.sleep", fake_sleep):
            await limiter.throttle()

    assert len(slept) == 1
    assert slept[0] == pytest.approx(0.5, abs=0.01)  # 1 token / 2 per sec


@pytest.mark.asyncio
async def test_spinner_skips_edit_when_rate_limited():
    """Spinner does not call edit_text when no tokens are available."""
    msg = MagicMock()
    msg.edit_text = AsyncMock()

    exhausted = _TelegramRateLimiter(rate=2.0, capacity=1.0)
    exhausted._tokens = 0.0

    sleep_calls = [0]

    async def fake_sleep(s):
        sleep_calls[0] += 1
        if sleep_calls[0] >= 2:
            raise asyncio.CancelledError()

    with (
        patch("aug.api.interfaces.telegram.interface._rate_limiter", exhausted),
        patch("aug.api.interfaces.telegram.interface.asyncio.sleep", fake_sleep),
    ):
        try:
            await _spinner_task(msg, "brave_search", {"query": "q"}, [""])
        except asyncio.CancelledError:
            pass

    msg.edit_text.assert_not_called()


@pytest.mark.asyncio
async def test_tool_end_uses_throttle_not_try_acquire(interface):
    """ToolEndEvent final status uses throttle() (must-deliver), not try_acquire() (droppable)."""
    from aug.api.interfaces.telegram.interface import TelegramInterface
    from aug.core.events import ToolEndEvent, ToolStartEvent

    context, _msg, _bot, _sent = _make_context()

    mock_limiter = MagicMock(spec=_TelegramRateLimiter)
    mock_limiter.throttle = AsyncMock()
    mock_limiter.try_acquire = MagicMock(return_value=True)

    with (
        patch("aug.api.interfaces.telegram.interface._rate_limiter", mock_limiter),
        patch("aug.api.interfaces.telegram.interface._typing_loop", AsyncMock()),
    ):
        iface = TelegramInterface(checkpointer=MagicMock())
        await iface.send_stream(
            _stream(
                ToolStartEvent(
                    run_id="r1", tool_name="brave_search", args={"query": "q"}, parent_ids=[]
                ),
                ToolEndEvent(run_id="r1", tool_name="brave_search", output=None, error=False),
            ),
            context,
        )

    # throttle() must have been called at least once (for the final ✓ edit)
    mock_limiter.throttle.assert_called()
    # The final ✓ edit must NOT go through try_acquire (which could drop it)
    tool_msg = _sent[0]
    tool_msg.edit_text.assert_called_once()


@pytest.mark.asyncio
async def test_reply_text_uses_throttle():
    """_reply_text_with_retry goes through throttle() so final messages are always delivered."""
    from aug.api.interfaces.telegram.interface import TelegramInterface
    from aug.core.events import ChatModelStreamEvent

    context, msg, _bot, _sent = _make_context()

    mock_limiter = MagicMock(spec=_TelegramRateLimiter)
    mock_limiter.throttle = AsyncMock()
    mock_limiter.try_acquire = MagicMock(return_value=True)

    with (
        patch("aug.api.interfaces.telegram.interface._rate_limiter", mock_limiter),
        patch("aug.api.interfaces.telegram.interface._typing_loop", AsyncMock()),
    ):
        iface = TelegramInterface(checkpointer=MagicMock())
        await iface.send_stream(
            _stream(ChatModelStreamEvent(delta="Hello")),
            context,
        )

    # throttle() called at least once covering the final reply_text
    mock_limiter.throttle.assert_called()
    msg.reply_text.assert_called()
