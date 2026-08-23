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
import re
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from aug.core.dispatch import TASK_RETRY_JOB_PREFIX, fire_task
from aug.utils.job_control import set_fire_task_fn, set_scheduler
from aug.utils.tasks import ScheduledTask, list_tasks, make_trigger

logger = logging.getLogger(__name__)

_RECONCILE_INTERVAL = 30  # seconds
# Missed jobs older than this are silently dropped instead of "caught up".
_MISFIRE_GRACE_SECONDS = 3600


async def start_scheduler(app: FastAPI) -> asyncio.Task:
    """Create, start, and return an APScheduler backed by Postgres.

    Performs an initial reconciliation to register all enabled tasks, then
    launches a background asyncio.Task that re-syncs every
    ``_RECONCILE_INTERVAL`` seconds.

    The task should be cancelled during application shutdown.
    """
    scheduler = AsyncIOScheduler(
        timezone=UTC,
        job_defaults={"misfire_grace_time": _MISFIRE_GRACE_SECONDS},
    )
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
        if not task.enabled:
            continue
        job_id = task.id
        wanted_ids.add(job_id)
        schedule_key = f"{task.schedule_type}:{json.dumps(task.schedule_params, sort_keys=True)}"

        # Skip one-shot date tasks that have already fired.
        if _is_date_task_past(task):
            if job_id in current_job_ids:
                scheduler.remove_job(job_id)
                schedule_cache.pop(job_id, None)
            wanted_ids.discard(job_id)
            logger.debug("Skipping past date task %r", task.name)
            continue

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


def _is_date_task_past(task: ScheduledTask) -> bool:
    """Return True if a one-shot date task has already passed."""
    if task.schedule_type != "date":
        return False
    run_date_raw = task.schedule_params.get("run_date")
    if not run_date_raw:
        return False
    try:
        if isinstance(run_date_raw, datetime):
            run_date = run_date_raw
        elif isinstance(run_date_raw, str):
            # Normalise PostgreSQL text format: space → T, +HH → +HH:00
            s = run_date_raw.replace(" ", "T")
            s = re.sub(r"([+-]\d{2})$", r"\1:00", s)
            run_date = datetime.fromisoformat(s)
        else:
            return False
        # Compare against now in the task's timezone (or UTC if naive)
        if run_date.tzinfo is None:
            run_date = run_date.replace(tzinfo=UTC)
        return run_date < datetime.now(UTC)
    except Exception:
        logger.warning("Could not parse run_date %r for task %r", run_date_raw, task.name)
        return False
