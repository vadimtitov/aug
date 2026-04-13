"""Unit tests for the set_reminder tool."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

_CONFIG = {"configurable": {"thread_id": "test-thread", "interface": "", "sender_id": ""}}


def _future_iso(hours: float = 2.0) -> str:
    return (datetime.now(tz=UTC) + timedelta(hours=hours)).isoformat()


def _invoke(when: str, message: str):
    from aug.core.tools.set_reminder import set_reminder

    return set_reminder.ainvoke({"when": when, "message": message}, config=_CONFIG)


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
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="uuid-1234")
    mock_conn.close = AsyncMock()

    with patch("aug.core.tools.set_reminder.asyncpg.connect", AsyncMock(return_value=mock_conn)):
        result = await _invoke(_future_iso(2), "call dentist")

    assert "call dentist" in result
    mock_conn.fetchval.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_set_reminder_db_error():
    import asyncpg

    with patch(
        "aug.core.tools.set_reminder.asyncpg.connect",
        AsyncMock(side_effect=asyncpg.PostgresError("conn refused")),
    ):
        result = await _invoke(_future_iso(2), "test")

    assert "failed" in result.lower()


@pytest.mark.asyncio
async def test_set_reminder_naive_datetime_accepted():
    """Naive datetimes should be accepted and treated as UTC."""
    naive = (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="uuid-9999")
    mock_conn.close = AsyncMock()

    with patch("aug.core.tools.set_reminder.asyncpg.connect", AsyncMock(return_value=mock_conn)):
        result = await _invoke(naive, "naive test")

    assert "naive test" in result


@pytest.mark.asyncio
async def test_set_reminder_stores_notification_target():
    """Notification interface and target from config should be stored on the reminder."""
    from aug.core.tools.set_reminder import set_reminder

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="uuid-5678")
    mock_conn.close = AsyncMock()

    config = {
        "configurable": {
            "thread_id": "test-thread",
            "interface": "telegram",
            "sender_id": "999888777",
        }
    }

    with patch("aug.core.tools.set_reminder.asyncpg.connect", AsyncMock(return_value=mock_conn)):
        result = await set_reminder.ainvoke(
            {"when": _future_iso(1), "message": "buy milk"}, config=config
        )

    assert "buy milk" in result
    call_args = mock_conn.fetchval.call_args
    assert "telegram" in call_args.args
    assert "999888777" in call_args.args
