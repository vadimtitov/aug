"""Unit tests for the set_reminder tool."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_CONFIG = {"configurable": {"thread_id": "test-thread", "interface": "telegram", "sender_id": ""}}


def _future_iso(hours: float = 2.0) -> str:
    return (datetime.now(tz=UTC) + timedelta(hours=hours)).isoformat()


def _invoke(when: str, message: str, config=None):
    from aug.core.tools.set_reminder import set_reminder

    return set_reminder.ainvoke({"when": when, "message": message}, config=config or _CONFIG)


def _make_pool_mock(task_id: str = "uuid-1234"):
    """Return a mock pool whose acquire() context manager returns a conn with fetchval."""
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=task_id)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_pool, mock_conn


@pytest.mark.asyncio
async def test_set_reminder_invalid_datetime():
    result = await _invoke("not-a-date", "hello")
    assert "invalid" in result.lower()


@pytest.mark.asyncio
async def test_set_reminder_past_datetime():
    past = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    result = await _invoke(past, "too late")
    assert "past" in result.lower()


@pytest.mark.asyncio
async def test_set_reminder_success():
    mock_pool, mock_conn = _make_pool_mock()

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        result = await _invoke(_future_iso(2), "call dentist")

    assert "call dentist" in result
    mock_conn.fetchval.assert_called_once()


@pytest.mark.asyncio
async def test_set_reminder_db_error():
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(side_effect=Exception("conn refused"))
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        result = await _invoke(_future_iso(2), "test")

    assert "failed" in result.lower()


@pytest.mark.asyncio
async def test_set_reminder_naive_datetime_accepted():
    """Naive datetimes should be accepted and treated as UTC."""
    naive = (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
    mock_pool, _ = _make_pool_mock()

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        result = await _invoke(naive, "naive test")

    assert "naive test" in result


@pytest.mark.asyncio
async def test_set_reminder_stores_push_type_forward():
    """Reminder tasks must be created with push_type='forward'."""
    from aug.core.tools.set_reminder import set_reminder

    mock_pool, mock_conn = _make_pool_mock()
    config = {
        "configurable": {
            "thread_id": "test-thread",
            "interface": "telegram",
            "sender_id": "999888777",
        }
    }

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        result = await set_reminder.ainvoke(
            {"when": _future_iso(1), "message": "buy milk"}, config=config
        )

    assert "buy milk" in result
    call_args = mock_conn.fetchval.call_args
    # push_type='forward' must appear in the INSERT args
    assert "forward" in call_args.args


@pytest.mark.asyncio
async def test_set_reminder_thread_id_uses_sender_id():
    """Thread ID should embed the sender_id for deterministic Telegram routing."""
    from aug.core.tools.set_reminder import set_reminder

    mock_pool, mock_conn = _make_pool_mock()
    config = {
        "configurable": {
            "thread_id": "tg-123-1",
            "interface": "telegram",
            "sender_id": "123",
        }
    }

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        await set_reminder.ainvoke({"when": _future_iso(1), "message": "ping"}, config=config)

    call_args = mock_conn.fetchval.call_args
    # thread_id 'default:123' must appear in the INSERT args
    assert "default:123" in call_args.args


@pytest.mark.asyncio
async def test_set_reminder_message_has_clock_prefix():
    """Stored message must have the ⏰ prefix."""
    mock_pool, mock_conn = _make_pool_mock()

    with patch("aug.core.tools.set_reminder.get_pool", return_value=mock_pool):
        await _invoke(_future_iso(1), "take meds")

    call_args = mock_conn.fetchval.call_args
    stored_message = next(a for a in call_args.args if isinstance(a, str) and "take meds" in a)
    assert stored_message.startswith("⏰")
