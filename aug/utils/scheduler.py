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
from datetime import UTC

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from aug.core.dispatch import TASK_RETRY_JOB_PREFIX, fire_task
from aug.utils.job_control import set_fire_task_fn, set_scheduler
from aug.utils.tasks import list_tasks, make_trigger

logger = logging.getLogger(__name__)

_RECONCILE_INTERVAL = 30  # seconds


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
        if not task.enabled:
            continue
        job_id = task.id
        wanted_ids.add(job_id)
        schedule_key = f"{task.schedule_type}:{json.dumps(task.schedule_params, sort_keys=True)}"

        if job_id in current_job_ids:
            if schedule_cache.get(job_id) != schedule_key:
                trigger = make_trigger(task.schedule_type, task.schedule_params)
                scheduler.reschedule_job(job_id, trigger=trigger)
                schedule_cache[job_id] = schedule_key
        else:
            scheduler.add_job(
                fire_task,
                trigger=make_trigger(task.schedule_type, task.schedule_params),
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


