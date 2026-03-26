"""Unit tests for the reflex infrastructure."""

import asyncio
import time
from unittest.mock import AsyncMock

from aug.core.reflexes import Reflex, ReflexOutput, run_reflexes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reflex(output: ReflexOutput | None) -> Reflex:
    return AsyncMock(return_value=output)


# ---------------------------------------------------------------------------
# run_reflexes — basic behaviour
# ---------------------------------------------------------------------------


async def test_run_reflexes_empty_list_returns_empty() -> None:
    assert await run_reflexes([], "hello", []) == []


async def test_run_reflexes_returns_non_none_outputs() -> None:
    reflexes = [
        _make_reflex(ReflexOutput(inject="did thing A", display="A done")),
        _make_reflex(None),  # skipped
        _make_reflex(ReflexOutput(inject="did thing B")),
    ]
    results = await run_reflexes(reflexes, "query", [])
    assert len(results) == 2
    assert results[0].inject == "did thing A"
    assert results[0].display == "A done"
    assert results[1].inject == "did thing B"
    assert results[1].display is None


async def test_run_reflexes_passes_query_and_history() -> None:
    captured: dict = {}

    async def spy_reflex(query: str, history: list) -> ReflexOutput | None:
        captured["query"] = query
        captured["history"] = history
        return None

    history = ["previous message"]
    await run_reflexes([spy_reflex], "turn on lights", history)

    assert captured["query"] == "turn on lights"
    assert captured["history"] is history


# ---------------------------------------------------------------------------
# run_reflexes — error isolation
# ---------------------------------------------------------------------------


async def test_run_reflexes_swallows_exception_from_one_reflex() -> None:
    async def bad_reflex(query: str, history: list) -> ReflexOutput | None:
        raise RuntimeError("boom")

    good = _make_reflex(ReflexOutput(inject="ok"))
    results = await run_reflexes([bad_reflex, good], "query", [])
    assert len(results) == 1
    assert results[0].inject == "ok"


async def test_run_reflexes_swallows_timeout() -> None:
    async def slow_reflex(query: str, history: list) -> ReflexOutput | None:
        await asyncio.sleep(60)
        return ReflexOutput(inject="never")

    good = _make_reflex(ReflexOutput(inject="fast"))
    results = await run_reflexes([slow_reflex, good], "query", [], reflex_timeout=0.05)
    assert len(results) == 1
    assert results[0].inject == "fast"


# ---------------------------------------------------------------------------
# run_reflexes — all run in parallel
# ---------------------------------------------------------------------------


async def test_run_reflexes_runs_in_parallel() -> None:
    """All reflexes should overlap — total time should be close to one reflex duration."""
    delay = 0.05
    results_order: list[int] = []

    async def make_reflex(i: int) -> Reflex:
        async def r(query: str, history: list) -> ReflexOutput | None:
            await asyncio.sleep(delay)
            results_order.append(i)
            return ReflexOutput(inject=f"reflex {i}")

        return r

    reflexes = [await make_reflex(i) for i in range(3)]

    t0 = time.monotonic()
    results = await run_reflexes(reflexes, "query", [])
    elapsed = time.monotonic() - t0

    assert len(results) == 3
    # All three ran, but total time is ~delay, not 3*delay
    assert elapsed < delay * 2
