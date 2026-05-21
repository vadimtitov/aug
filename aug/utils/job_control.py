"""Thin scheduler-control singleton for use by agent tools.

Kept separate from scheduler.py so tools can cancel jobs without pulling in
aug.core (which would create a circular import through registry).

Usage:
    # At startup (in scheduler.py):
    set_scheduler(scheduler)
    set_fire_task_fn(fire_task)

    # In tools or anywhere else:
    cancel_job(job_id)
    add_task_job(task_id, trigger)
"""

import logging
from collections.abc import Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None
_fire_task_fn: Callable | None = None


def set_scheduler(scheduler: AsyncIOScheduler) -> None:
    global _scheduler
    _scheduler = scheduler


def set_fire_task_fn(fn: Callable) -> None:
    global _fire_task_fn
    _fire_task_fn = fn


def add_task_job(task_id: str, trigger: Any) -> None:
    """Register a new job with APScheduler immediately.

    Called right after DB insert so short-duration tasks (e.g. 10-second reminders)
    don't have to wait for the next 30-second reconciler cycle.
    Best-effort; the reconciler will catch it on the next pass if this fails.
    """
    if _scheduler is None or _fire_task_fn is None:
        logger.warning("add_task_job called before scheduler/fire_task_fn set — skipping")
        return
    try:
        _scheduler.add_job(
            _fire_task_fn,
            trigger=trigger,
            id=task_id,
            args=[task_id],
            replace_existing=True,
        )
    except Exception:
        logger.exception("add_task_job failed for task_id=%s", task_id)


def cancel_job(job_id: str) -> None:
    """Remove a job from APScheduler immediately. Best-effort; safe if not present."""
    if _scheduler is None:
        return
    try:
        _scheduler.remove_job(job_id)
    except Exception:
        pass
