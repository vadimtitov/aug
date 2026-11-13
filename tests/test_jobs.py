"""Background jobs: the global list, scheduling, catch-up and manual runs."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from aug.core.jobs import _JOB_FUNCTIONS, JOBS, JobRunner, JobSpec, _is_due
from aug.core.registry import list_agents
from aug.utils.state import AppState

_DAILY = JobSpec(job="daily", cron="0 3 * * *", profile="p1")


class _State:
    """state.json in memory."""

    def __init__(self, jobs: dict[str, str] | None = None) -> None:
        self.state = AppState(jobs=jobs or {})

    def load(self) -> AppState:
        return self.state.model_copy(deep=True)

    def save(self, state: AppState) -> None:
        self.state = state


@pytest.fixture
def state():
    fake = _State()
    with (
        patch("aug.core.jobs.load_state", side_effect=fake.load),
        patch("aug.core.jobs.save_state", side_effect=fake.save),
    ):
        yield fake


def _runner(jobs, calls):
    async def job(harness):
        calls.append(harness)
        return "result"

    return JobRunner(
        jobs=jobs,
        functions={spec.job: job for spec in jobs},
        harness_for=lambda name: f"harness:{name}",
    )


def test_every_listed_job_exists_and_runs_on_a_registered_profile():
    assert {spec.job for spec in JOBS} == set(_JOB_FUNCTIONS)
    assert all(spec.profile in list_agents() for spec in JOBS)
    assert len({spec.job for spec in JOBS}) == len(JOBS)


async def test_each_job_is_scheduled_and_caught_up_once_with_two_profiles_live(state):
    calls: list[str] = []
    other = JobSpec(job="weekly", cron="0 4 * * sun", profile="p2")
    runner = _runner([_DAILY, other], calls)
    await runner.start()
    try:
        await asyncio.sleep(0)  # the catch-up runs the never-run job
        await runner._catch_up
        assert sorted(runner.scheduled()) == ["daily", "weekly"]
        assert calls == ["harness:p1", "harness:p2"]
    finally:
        await runner.stop()


async def test_a_job_run_since_its_last_scheduled_time_is_not_caught_up(state):
    state.state.jobs["daily"] = datetime.now(UTC).isoformat()
    calls: list[str] = []
    runner = _runner([_DAILY], calls)
    await runner.start()
    try:
        await runner._catch_up
        assert calls == []
    finally:
        await runner.stop()


async def test_a_manual_run_always_runs_and_records_it(state):
    state.state.jobs["daily"] = datetime.now(UTC).isoformat()
    calls: list[str] = []
    runner = _runner([_DAILY], calls)

    assert await runner.run("daily") == "result"

    assert calls == ["harness:p1"]
    assert datetime.fromisoformat(state.state.jobs["daily"]) <= datetime.now(UTC)


async def test_a_manual_run_of_an_unknown_job_raises(state):
    with pytest.raises(ValueError, match="nope"):
        await _runner([_DAILY], []).run("nope")


async def test_a_failed_scheduled_run_is_not_recorded_and_does_not_raise(state):
    async def broken(harness):
        raise RuntimeError("model down")

    runner = JobRunner(jobs=[_DAILY], functions={"daily": broken}, harness_for=str)
    await runner._run_if_due("daily")

    assert "daily" not in state.state.jobs


async def test_no_two_job_runs_overlap(state):
    running = 0
    peak = 0

    async def slow(harness):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    weekly = JobSpec(job="weekly", cron="0 4 * * sun", profile="p1")
    runner = JobRunner(
        jobs=[_DAILY, weekly], functions={"daily": slow, "weekly": slow}, harness_for=str
    )
    await asyncio.gather(runner.run("daily"), runner._run_if_due("daily"), runner.run("weekly"))

    assert peak == 1


@pytest.mark.parametrize(
    ("last_run", "now", "due"),
    [
        (None, "2026-11-13T12:00:00", True),
        ("2026-11-13T03:00:05", "2026-11-13T12:00:00", False),  # ran at today's 03:00
        ("2026-11-12T03:00:05", "2026-11-13T02:59:00", False),  # today's 03:00 not yet
        ("2026-11-12T03:00:05", "2026-11-13T03:30:00", True),  # missed today's
        ("2026-11-10T03:00:05", "2026-11-13T01:00:00", True),  # down for days
    ],
)
def test_is_due(last_run, now, due):
    def parse(iso):
        return datetime.fromisoformat(iso).replace(tzinfo=UTC) if iso else None

    assert _is_due(_DAILY, parse(last_run), parse(now)) is due


def test_the_deep_dream_is_due_once_a_week():
    deep = next(spec for spec in JOBS if spec.job == "dream_deep")
    saturday = datetime(2026, 11, 14, 12, tzinfo=UTC)
    after_sunday = datetime(2026, 11, 15, 5, tzinfo=UTC)
    last = datetime(2026, 11, 8, 4, 0, 5, tzinfo=UTC)  # last Sunday's run

    assert _is_due(deep, last, saturday) is False
    assert _is_due(deep, last, after_sunday) is True
