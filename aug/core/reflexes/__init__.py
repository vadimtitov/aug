"""Reflex infrastructure — fast parallel actions that fire alongside the main agent.

A Reflex is a lightweight async function that receives the user's query and
recent history, optionally acts immediately (e.g. executes a Home Assistant
command), and returns a ReflexOutput describing what it did.

The output has two parts:
  inject  — injected into the agent conversation so the main agent knows what
             happened and can incorporate it into its response.
  display — shown directly to the user as a quick confirmation (None to skip).

Reflexes run in parallel with the main agent via asyncio.create_task.  If the
reflex completes while the agent is still running the inject is picked up at
the next interrupt_after=["call_tools"] pause point.  If the agent finishes
first the inject lands in the leftover queue and triggers a follow-up run.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aug.core.run import MessageContent

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


@dataclass
class ReflexOutput:
    """Result produced by a Reflex."""

    inject: str
    """Text injected into the agent conversation to describe what happened."""

    display: str | None = None
    """Quick confirmation shown directly to the user.  None to skip."""


Reflex = Callable[[str, list[MessageContent]], Awaitable[ReflexOutput | None]]
"""
async def my_reflex(query: str, history: list[MessageContent]) -> ReflexOutput | None:
    ...

Return None if the reflex decides it has nothing to do for this query.
"""


async def run_reflexes(
    reflexes: list[Reflex],
    query: str,
    history: list[MessageContent],
    reflex_timeout: float = _DEFAULT_TIMEOUT,
) -> list[ReflexOutput]:
    """Run all reflexes in parallel and return the non-None results.

    Each reflex is given ``reflex_timeout`` seconds.  Timeouts and exceptions
    are swallowed and logged — a misbehaving reflex must never crash the agent.
    """
    if not reflexes:
        return []

    async def _run_one(reflex: Reflex) -> ReflexOutput | None:
        name = getattr(reflex, "__name__", repr(reflex))
        logger.info("reflex_start reflex=%s", name)
        try:
            result = await asyncio.wait_for(reflex(query, history), timeout=reflex_timeout)
            if result is None:
                logger.info("reflex_skip reflex=%s", name)
            else:
                logger.info(
                    "reflex_done reflex=%s inject=%.80r display=%.40r",
                    name,
                    result.inject,
                    result.display,
                )
            return result
        except TimeoutError:
            logger.warning("reflex_timeout reflex=%s timeout=%.1fs", name, reflex_timeout)
        except Exception:
            logger.exception("reflex_error reflex=%s", name)
        return None

    results = await asyncio.gather(*(_run_one(r) for r in reflexes))
    return [r for r in results if r is not None]
