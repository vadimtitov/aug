"""Background jobs: async functions that receive a harness.

A job is registered by name in ``_JOB_FUNCTIONS``.  ``JOBS`` is the one list that says
which jobs run, when (a UTC crontab) and on which profile (agent version), so each job
runs once however many agent versions are live.  A job does its model work through
``harness.oneshot()`` and asks for a tier, never a model.

``job_runner`` schedules the list at startup.  A run that was due while AUG was down
happens once at startup; a manual run (``/consolidate``) goes through the same runner.
One job runs at a time: the dreams write the same files.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from aug.core.harness.base import Harness
from aug.core.memory import run_deep_consolidation, run_light_consolidation
from aug.core.registry import get_harness
from aug.utils.state import load_state, save_state

logger = logging.getLogger(__name__)

Job = Callable[[Harness], Awaitable[object]]


@dataclass(frozen=True)
class JobSpec:
    job: str  # a name in _JOB_FUNCTIONS
    cron: str  # crontab, UTC
    profile: str  # the agent version whose harness runs it


JOBS: list[JobSpec] = [
    JobSpec(job="dream_light", cron="0 3 * * *", profile="v13_kimi"),
    JobSpec(job="dream_deep", cron="0 4 * * sun", profile="v13_kimi"),
]

_JOB_FUNCTIONS: dict[str, Job] = {
    "dream_light": run_light_consolidation,
    "dream_deep": run_deep_consolidation,
}


class JobRunner:
    """Runs JOBS on their schedule, one job at a time."""

    def __init__(
        self,
        jobs: list[JobSpec] = JOBS,
        functions: dict[str, Job] = _JOB_FUNCTIONS,
        harness_for: Callable[[str], Harness] = get_harness,
    ) -> None:
        self._jobs = {spec.job: spec for spec in jobs}
        self._functions = functions
        self._harness_for = harness_for
        self._lock = asyncio.Lock()
        self._scheduler: AsyncIOScheduler | None = None
        self._catch_up: asyncio.Task | None = None

    async def start(self) -> None:
        """Schedule every job, and run the ones that came due while AUG was down
        (in the background, so startup never waits on a model)."""
        scheduler = AsyncIOScheduler(timezone=UTC)
        for spec in self._jobs.values():
            scheduler.add_job(
                self._run_if_due,
                _trigger(spec),
                args=[spec.job],
                id=spec.job,
                coalesce=True,
                misfire_grace_time=None,  # however late (a suspended host), still run
            )
        scheduler.start()
        self._scheduler = scheduler
        self._catch_up = asyncio.create_task(self._run_missed(), name="jobs-catch-up")

    async def stop(self) -> None:
        if self._catch_up is not None:
            self._catch_up.cancel()
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    async def run(self, name: str) -> object:
        """Run job *name* now, whether or not it is due.  Returns what the job returns;
        raises what it raises."""
        spec = self._jobs.get(name)
        if spec is None:
            raise ValueError(f"Unknown job {name!r}")
        async with self._lock:
            return await self._execute(spec)

    def scheduled(self) -> list[str]:
        """Names of the jobs the scheduler holds."""
        return [job.id for job in self._scheduler.get_jobs()] if self._scheduler else []

    async def _run_missed(self) -> None:
        for name in self._jobs:
            await self._run_if_due(name)

    async def _run_if_due(self, name: str) -> None:
        """Run *name* if a scheduled time has passed since its last run.  Never raises:
        a failed run is retried at the next schedule or startup."""
        spec = self._jobs[name]
        async with self._lock:
            if not _is_due(spec, _last_run(name), datetime.now(UTC)):
                return
            try:
                await self._execute(spec)
            except Exception:
                logger.exception("job %s failed, will retry at its next run", name)

    async def _execute(self, spec: JobSpec) -> object:
        logger.info("job %s starting on %s", spec.job, spec.profile)
        result = await self._functions[spec.job](self._harness_for(spec.profile))
        _record_run(spec.job)
        logger.info("job %s done", spec.job)
        return result


job_runner = JobRunner()


def _trigger(spec: JobSpec) -> CronTrigger:
    return CronTrigger.from_crontab(spec.cron, timezone=UTC)


def _is_due(spec: JobSpec, last_run: datetime | None, now: datetime) -> bool:
    """True when the job never ran, or a scheduled time has passed since it last did."""
    if last_run is None:
        return True
    upcoming = _trigger(spec).get_next_fire_time(None, last_run)
    return upcoming is not None and upcoming <= now


def _last_run(name: str) -> datetime | None:
    iso = load_state().jobs.get(name)
    return datetime.fromisoformat(iso) if iso else None


def _record_run(name: str) -> None:
    state = load_state()
    state.jobs[name] = datetime.now(UTC).isoformat()
    save_state(state)
