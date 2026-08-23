"""Tests for scheduler reconciliation and date-task cleanup."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from aug.utils.scheduler import _is_date_task_past
from aug.utils.tasks import ScheduledTask


def make_task(schedule_type: str, schedule_params: dict, enabled: bool = True) -> ScheduledTask:
    return ScheduledTask(
        id="test-id",
        name="test-task",
        interface="telegram",
        thread_id="default",
        message="hello",
        schedule_type=schedule_type,
        schedule_params=schedule_params,
        enabled=enabled,
        push_type="agent",
        created_at=datetime.now(UTC),
    )


def test_is_date_task_past_with_past_date():
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    task = make_task("date", {"run_date": past})
    assert _is_date_task_past(task) is True


def test_is_date_task_past_with_future_date():
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    task = make_task("date", {"run_date": future})
    assert _is_date_task_past(task) is False


def test_is_date_task_past_non_date_type():
    task = make_task("interval", {"minutes": 30})
    assert _is_date_task_past(task) is False


def test_is_date_task_past_with_naive_datetime():
    past = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
    task = make_task("date", {"run_date": past})
    assert _is_date_task_past(task) is True


def test_is_date_task_past_with_postgres_text_format():
    """PostgreSQL returns timestamps as '2026-05-21 23:38:00+00'."""
    past = "2026-05-21 23:38:00+00"
    task = make_task("date", {"run_date": past})
    assert _is_date_task_past(task) is True


def test_is_date_task_past_missing_run_date():
    task = make_task("date", {})
    assert _is_date_task_past(task) is False


def test_is_date_task_past_disabled_task():
    """Disabled tasks are handled by the caller, not _is_date_task_past."""
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    task = make_task("date", {"run_date": past}, enabled=False)
    # Function itself doesn't check enabled; caller does
    assert _is_date_task_past(task) is True
