"""APScheduler integration — timing engine for scheduled tasks.

Uses an in-memory job store; ``scheduled_tasks`` Postgres table is the
authoritative store.  On startup the reconciler loads all enabled tasks and
registers them with APScheduler.  A background reconciler re-reads the table
every 30 seconds to pick up changes made by agent tools (create, update, delete).

The scheduler is stored on ``app.state.scheduler`` for the reconciler loop
and for graceful shutdown.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import asyncpg
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from aug.core.dispatch import TASK_RETRY_JOB_PREFIX, fire_task
from aug.utils.job_control import set_fire_task_fn, set_scheduler
from aug.utils.tasks import (
    ScheduledTask,
    list_tasks,
    make_trigger,
    mark_fired,
    normalize_run_date,
)

logger = logging.getLogger(__name__)

_RECONCILE_INTERVAL = 30  # seconds
# How far past its run_date an unfired one-shot must be before the reconciler writes
# it off as missed.  Only needs to cover the gap between a job coming due and
# APScheduler dispatching it — a busy event loop can stretch that — so a reconcile
# tick landing in the middle can never write off a job that is about to run.
_MISSED_ONE_SHOT_AFTER = timedelta(minutes=5)


async def start_scheduler(app: FastAPI) -> asyncio.Task:
    """Create, start, and return an APScheduler backed by Postgres.

    Performs an initial reconciliation to register all enabled tasks, then
    launches a background asyncio.Task that re-syncs every
    ``_RECONCILE_INTERVAL`` seconds.

    The task should be cancelled during application shutdown.
    """
    scheduler = AsyncIOScheduler(timezone=UTC)
    scheduler.start()
    app.state.scheduler = scheduler
    set_scheduler(scheduler)
    set_fire_task_fn(fire_task)

    await _reconcile(app)
    return asyncio.create_task(_reconciler_loop(app), name="scheduler-reconciler")


async def stop_scheduler(app: FastAPI) -> None:
    """Gracefully shut down the APScheduler."""
    scheduler: AsyncIOScheduler | None = getattr(app.state, "scheduler", None)
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")


async def _reconciler_loop(app: FastAPI) -> None:
    try:
        while True:
            await asyncio.sleep(_RECONCILE_INTERVAL)
            try:
                await _reconcile(app)
            except Exception:
                logger.exception("Scheduler reconciliation failed")
    except asyncio.CancelledError:
        logger.info("Scheduler reconciler shut down cleanly.")


async def _reconcile(app: FastAPI) -> None:
    """Sync APScheduler jobs with the current state of the scheduled_tasks table."""
    pool = app.state.db_pool
    scheduler: AsyncIOScheduler = app.state.scheduler
    # Keyed by job_id; value is "schedule_type:json(schedule_params)".
    # Only reschedule when this changes so interval triggers are not reset every 30 s.
    schedule_cache: dict[str, str] = getattr(app.state, "_scheduler_cache", {})

    async with pool.acquire() as conn:
        tasks = await list_tasks(conn)

    current_job_ids = {job.id for job in scheduler.get_jobs()}
    wanted_ids: set[str] = set()

    for task in tasks:
        if not task.enabled or task.fired_at is not None:
            continue
        if _is_missed_one_shot(task):
            await _write_off(pool, task)
            continue

        job_id = task.id
        wanted_ids.add(job_id)
        schedule_key = f"{task.schedule_type}:{json.dumps(task.schedule_params, sort_keys=True)}"

        try:
            trigger = make_trigger(task.schedule_type, task.schedule_params)
        except Exception:
            logger.exception(
                "Skipping task %r — bad schedule params: %s", task.name, task.schedule_params
            )
            wanted_ids.discard(job_id)
            continue

        if job_id in current_job_ids:
            if schedule_cache.get(job_id) != schedule_key:
                scheduler.reschedule_job(job_id, trigger=trigger)
                schedule_cache[job_id] = schedule_key
        else:
            scheduler.add_job(
                fire_task,
                trigger=trigger,
                id=job_id,
                args=[task.id],
                replace_existing=True,
            )
            schedule_cache[job_id] = schedule_key

    for job_id in current_job_ids - wanted_ids:
        if job_id.startswith(TASK_RETRY_JOB_PREFIX):
            continue  # retry jobs are self-managing; reconciler must not remove them
        scheduler.remove_job(job_id)
        schedule_cache.pop(job_id, None)

    app.state._scheduler_cache = schedule_cache
    logger.debug("scheduler_reconcile total=%d enabled=%d", len(tasks), len(wanted_ids))


def _is_missed_one_shot(task: ScheduledTask) -> bool:
    """Return True if *task* fires once, at a moment that passed without it firing.

    Only ever asked about tasks with no ``fired_at``, so a delivered one never gets
    here.  What is left is a one-shot the service was not running for, or one created
    after its own deadline: there is no run left in it either way.

    The margin keeps a job that is merely mid-dispatch — due, but not yet handed to
    the executor on a busy event loop — from being written off a moment too early.
    """
    if task.schedule_type != "date":
        return False
    run_date = task.schedule_params.get("run_date")
    if not run_date:
        return False
    try:
        parsed = normalize_run_date(run_date)
    except (TypeError, ValueError):
        # Leave it to make_trigger to reject and log with the full params.
        logger.warning("Task %r has an unparseable run_date: %r", task.name, run_date)
        return False
    # A naive run_date is localised to the scheduler's timezone, which is UTC.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed < datetime.now(UTC) - _MISSED_ONE_SHOT_AFTER


async def _write_off(pool: asyncpg.Pool, task: ScheduledTask) -> None:
    """Record a missed one-shot as finished, so it is judged once and not every pass.

    Says nothing about delivery — ``fire_task`` never reads this column, so a retry
    still in flight for this task runs and reports as usual.
    """
    logger.warning(
        "Task %r was never delivered — its run_date passed while nothing was scheduled",
        task.name,
    )
    try:
        async with pool.acquire() as conn:
            await mark_fired(conn, task.id)
    except Exception:
        logger.exception("Could not write off missed task %r", task.name)
