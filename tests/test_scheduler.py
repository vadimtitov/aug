"""Tests for one-shot task completion — the fired_at stamp and the reconciler.

A ``date`` task has no next run, but its row stays in ``scheduled_tasks``.  With no
way to record that it was over, the reconciler re-registered it with APScheduler
every 30 seconds, where it could only misfire — one warning per task per pass, for
ever.  ``fired_at`` is that record: ``fire_task`` stamps it on delivery, the
reconciler stamps a one-shot whose moment passed unfired, and it schedules only rows
where the stamp is absent.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.utils.scheduler import _is_missed_one_shot, _reconcile
from aug.utils.tasks import ScheduledTask


def make_task(
    schedule_type: str,
    schedule_params: dict,
    *,
    enabled: bool = True,
    fired_at: datetime | None = None,
    task_id: str = "task-1",
) -> ScheduledTask:
    return ScheduledTask(
        id=task_id,
        name="test-task",
        interface="telegram",
        thread_id="default",
        message="hello",
        schedule_type=schedule_type,
        schedule_params=schedule_params,
        enabled=enabled,
        fired_at=fired_at,
        push_type="agent",
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# _is_missed_one_shot
# ---------------------------------------------------------------------------


def test_long_past_date_is_skipped():
    past = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    assert _is_missed_one_shot(make_task("date", {"run_date": past})) is True


def test_future_date_is_scheduled():
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    assert _is_missed_one_shot(make_task("date", {"run_date": future})) is False


def test_just_due_date_is_still_scheduled():
    """A job that came due seconds ago may be mid-dispatch — never pull it."""
    just_now = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    assert _is_missed_one_shot(make_task("date", {"run_date": just_now})) is False


def test_recurring_tasks_are_never_skipped():
    assert _is_missed_one_shot(make_task("interval", {"minutes": 30})) is False
    assert _is_missed_one_shot(make_task("cron", {"hour": 9})) is False


def test_postgres_text_timestamp_is_understood():
    """asyncpg can hand back '2026-05-21 23:38:00+00' — space separator, no offset minutes."""
    assert _is_missed_one_shot(make_task("date", {"run_date": "2026-05-21 23:38:00+00"})) is True


def test_naive_datetime_is_read_as_utc():
    past = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
    assert _is_missed_one_shot(make_task("date", {"run_date": past})) is True


def test_missing_run_date_is_left_to_make_trigger():
    assert _is_missed_one_shot(make_task("date", {})) is False


def test_unparseable_run_date_is_left_to_make_trigger():
    assert _is_missed_one_shot(make_task("date", {"run_date": "next tuesday"})) is False


# ---------------------------------------------------------------------------
# _reconcile
# ---------------------------------------------------------------------------


def _app(tasks: list[ScheduledTask], jobs: list[str] | None = None) -> MagicMock:
    """Build an app whose DB returns *tasks* and whose scheduler holds *jobs*."""
    conn = MagicMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    scheduler = MagicMock()
    scheduler.get_jobs.return_value = [SimpleNamespace(id=j) for j in (jobs or [])]

    app = MagicMock()
    app.state.db_pool = pool
    app.state.scheduler = scheduler
    app.state._scheduler_cache = {}
    return app


@pytest.mark.asyncio
async def test_reconcile_ignores_a_task_that_already_fired():
    """The stamp alone keeps it out — no date arithmetic involved."""
    task = make_task("date", {"run_date": "2099-01-01T00:00:00+00:00"}, fired_at=datetime.now(UTC))
    app = _app([task])

    with (
        patch("aug.utils.scheduler.list_tasks", new=AsyncMock(return_value=[task])),
        patch("aug.utils.scheduler.mark_fired", new=AsyncMock()) as mark,
    ):
        await _reconcile(app)

    app.state.scheduler.add_job.assert_not_called()
    mark.assert_not_awaited()  # already terminal — nothing to write


@pytest.mark.asyncio
async def test_reconcile_writes_off_a_one_shot_whose_moment_passed_unfired():
    past = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    task = make_task("date", {"run_date": past})
    app = _app([task])

    with (
        patch("aug.utils.scheduler.list_tasks", new=AsyncMock(return_value=[task])),
        patch("aug.utils.scheduler.mark_fired", new=AsyncMock()) as mark,
    ):
        await _reconcile(app)

    app.state.scheduler.add_job.assert_not_called()
    assert mark.await_args.args[1] == task.id


@pytest.mark.asyncio
async def test_reconcile_removes_a_lingering_past_one_shot_job():
    """A job left over from before the write-off is cleaned up in the same pass."""
    past = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    task = make_task("date", {"run_date": past})
    app = _app([task], jobs=[task.id])

    with (
        patch("aug.utils.scheduler.list_tasks", new=AsyncMock(return_value=[task])),
        patch("aug.utils.scheduler.mark_fired", new=AsyncMock()),
    ):
        await _reconcile(app)

    app.state.scheduler.remove_job.assert_called_once_with(task.id)


@pytest.mark.asyncio
async def test_reconcile_survives_a_failed_write_off():
    """A DB failure must not take down the reconcile pass for every other task."""
    past = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    tasks = [
        make_task("date", {"run_date": past}, task_id="missed"),
        make_task("cron", {"hour": 9}, task_id="daily"),
    ]
    app = _app(tasks)

    with (
        patch("aug.utils.scheduler.list_tasks", new=AsyncMock(return_value=tasks)),
        patch("aug.utils.scheduler.mark_fired", new=AsyncMock(side_effect=OSError("db gone"))),
    ):
        await _reconcile(app)

    scheduled = {call.kwargs["id"] for call in app.state.scheduler.add_job.call_args_list}
    assert scheduled == {"daily"}


@pytest.mark.asyncio
async def test_reconcile_still_schedules_future_and_recurring_tasks():
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    tasks = [
        make_task("date", {"run_date": future}, task_id="one-shot"),
        make_task("cron", {"hour": 9}, task_id="daily"),
    ]
    app = _app(tasks)

    with (
        patch("aug.utils.scheduler.list_tasks", new=AsyncMock(return_value=tasks)),
        patch("aug.utils.scheduler.mark_fired", new=AsyncMock()) as mark,
    ):
        await _reconcile(app)

    scheduled = {call.kwargs["id"] for call in app.state.scheduler.add_job.call_args_list}
    assert scheduled == {"one-shot", "daily"}
    mark.assert_not_awaited()


# ---------------------------------------------------------------------------
# Completion after delivery
# ---------------------------------------------------------------------------


@pytest.fixture()
def fired():
    """Patch fire_task's collaborators; yield (mark_fired mock, fire_push mock)."""
    from aug.core import dispatch

    conn = MagicMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(dispatch, "_app", MagicMock()),
        patch("aug.core.dispatch.get_pool", return_value=pool),
        patch("aug.core.dispatch.fire_push", new=AsyncMock()) as fire_push,
        patch("aug.core.dispatch.mark_fired", new=AsyncMock()) as mark_fired,
    ):
        yield mark_fired, fire_push


@pytest.mark.asyncio
async def test_delivered_one_shot_is_stamped(fired):
    from aug.core.dispatch import fire_task

    mark_fired, _ = fired
    task = make_task("date", {"run_date": "2026-08-24T09:00:00+00:00"})

    with patch("aug.core.dispatch.get_task", new=AsyncMock(return_value=task)):
        await fire_task(task.id)

    assert mark_fired.await_args.args[1] == task.id


@pytest.mark.asyncio
async def test_delivered_recurring_task_is_not_stamped(fired):
    """cron and interval tasks never end, so they must never carry a fired_at."""
    from aug.core.dispatch import fire_task

    mark_fired, _ = fired
    task = make_task("cron", {"hour": 9})

    with patch("aug.core.dispatch.get_task", new=AsyncMock(return_value=task)):
        await fire_task(task.id)

    mark_fired.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_one_shot_is_not_stamped(fired):
    """A failed delivery stays pending so its retry still has something to run."""
    from aug.core.dispatch import fire_task

    mark_fired, fire_push = fired
    fire_push.side_effect = RuntimeError("Telegram down")
    task = make_task("date", {"run_date": "2026-08-24T09:00:00+00:00"})

    with (
        patch("aug.core.dispatch.get_task", new=AsyncMock(return_value=task)),
        patch("aug.core.dispatch._schedule_task_retry") as retry,
    ):
        await fire_task(task.id)

    mark_fired.assert_not_awaited()
    retry.assert_called_once()


@pytest.mark.asyncio
async def test_fire_task_runs_a_stamped_task_so_retries_still_work(fired):
    """fire_task reads enabled, never fired_at — a retry after a write-off must deliver."""
    from aug.core.dispatch import fire_task

    _, fire_push = fired
    task = make_task("date", {"run_date": "2026-08-24T09:00:00+00:00"}, fired_at=datetime.now(UTC))

    with patch("aug.core.dispatch.get_task", new=AsyncMock(return_value=task)):
        await fire_task(task.id, retry_count=1)

    fire_push.assert_awaited_once()


@pytest.mark.asyncio
async def test_stamp_failure_does_not_raise(fired):
    """The message is already delivered; a failed bookkeeping write must not escape."""
    from aug.core.dispatch import fire_task

    mark_fired, _ = fired
    mark_fired.side_effect = OSError("db gone")
    task = make_task("date", {"run_date": "2026-08-24T09:00:00+00:00"})

    with (
        patch("aug.core.dispatch.get_task", new=AsyncMock(return_value=task)),
        patch("aug.core.dispatch._schedule_task_retry") as retry,
    ):
        await fire_task(task.id)

    retry.assert_not_called()  # delivery succeeded — this is not a retryable failure
