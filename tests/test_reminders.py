"""Unit tests for aug.core.reminders."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.utils.reminders import _fire_due_reminders, _next_retry_delay

# ---------------------------------------------------------------------------
# _next_retry_delay
# ---------------------------------------------------------------------------


def test_retry_delay_first_attempt():
    assert _next_retry_delay(0) == timedelta(minutes=1)


def test_retry_delay_second_attempt():
    assert _next_retry_delay(1) == timedelta(minutes=2)


def test_retry_delay_capped_at_60():
    assert _next_retry_delay(10) == timedelta(minutes=60)
    assert _next_retry_delay(100) == timedelta(minutes=60)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_row(**kwargs):
    defaults = {
        "id": "uuid-1",
        "message": "test message",
        "notification_interface": "telegram",
        "notification_target": "123456",
        "retry_count": 0,
    }
    defaults.update(kwargs)
    return defaults


def _make_pool(rows):
    """Build a mock asyncpg pool that returns *rows* from the SELECT."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.execute = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool, conn


def _make_app(pool):
    app = MagicMock()
    app.state.db_pool = pool
    return app


# ---------------------------------------------------------------------------
# _fire_due_reminders — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fire_marks_fired_after_delivery():
    row = _make_row()
    pool, conn = _make_pool([row])
    app = _make_app(pool)

    with patch("aug.utils.reminders.send_notification", AsyncMock()) as mock_notify:
        await _fire_due_reminders(app)

    mock_notify.assert_awaited_once_with(app, "telegram", "123456", "\u23f0 test message")
    # fired=TRUE update must have been called
    fired_calls = [c for c in conn.execute.call_args_list if "fired = TRUE" in c.args[0]]
    assert len(fired_calls) == 1
    assert fired_calls[0].args[1] == "uuid-1"


@pytest.mark.asyncio
async def test_fire_no_retry_update_on_success():
    """On success there must be no retry_count/next_retry_at UPDATE."""
    row = _make_row()
    pool, conn = _make_pool([row])
    app = _make_app(pool)

    with patch("aug.utils.reminders.send_notification", AsyncMock()):
        await _fire_due_reminders(app)

    retry_calls = [c for c in conn.execute.call_args_list if "retry_count" in c.args[0]]
    assert retry_calls == []


# ---------------------------------------------------------------------------
# _fire_due_reminders — failure / retry path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fire_does_not_mark_fired_on_failure():
    row = _make_row()
    pool, conn = _make_pool([row])
    app = _make_app(pool)

    with patch(
        "aug.utils.reminders.send_notification", AsyncMock(side_effect=RuntimeError("no network"))
    ):
        await _fire_due_reminders(app)

    fired_calls = [c for c in conn.execute.call_args_list if "fired = TRUE" in c.args[0]]
    assert fired_calls == []


@pytest.mark.asyncio
async def test_fire_increments_retry_on_failure():
    row = _make_row(retry_count=2)
    pool, conn = _make_pool([row])
    app = _make_app(pool)

    with patch(
        "aug.utils.reminders.send_notification", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await _fire_due_reminders(app)

    retry_calls = [c for c in conn.execute.call_args_list if "retry_count" in c.args[0]]
    assert len(retry_calls) == 1
    # positional args after SQL: $1=next_retry_at, $2=last_error, $3=id
    args = retry_calls[0].args
    assert args[3] == "uuid-1"
    assert "boom" in args[2]  # last_error


@pytest.mark.asyncio
async def test_fire_backoff_grows_with_retry_count():
    row0 = _make_row(id="a", retry_count=0)
    row3 = _make_row(id="b", retry_count=3)
    pool, conn = _make_pool([row0, row3])
    app = _make_app(pool)

    async def _fail_and_capture(*a, **kw):
        raise RuntimeError("fail")

    with patch("aug.utils.reminders.send_notification", AsyncMock(side_effect=_fail_and_capture)):
        await _fire_due_reminders(app)

    retry_calls = [c for c in conn.execute.call_args_list if "retry_count" in c.args[0]]
    assert len(retry_calls) == 2

    # next_retry_at is the first positional arg after the SQL string
    t0 = retry_calls[0].args[1]  # after retry_count=0 → +1 min
    t1 = retry_calls[1].args[1]  # after retry_count=3 → +8 min

    # t1 must be later than t0
    assert t1 > t0


@pytest.mark.asyncio
async def test_fire_no_rows_does_nothing():
    pool, conn = _make_pool([])
    app = _make_app(pool)

    with patch("aug.utils.reminders.send_notification", AsyncMock()) as mock_notify:
        await _fire_due_reminders(app)

    mock_notify.assert_not_awaited()
    conn.execute.assert_not_awaited()
