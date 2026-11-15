"""Reflexes: fast actions taken on what the user just said, before the agent's turn.

A Reflex gets the user's message and the conversation's last few exchanges, may act at
once (e.g. switch a light through Home Assistant), and returns a ReflexOutput saying
what it did, or None when it has nothing to do.  ``Harness.run()`` runs a profile's
reflexes on every turn someone is there for, before the agent starts:

  note    — added to the user's message, so the agent knows it is already done and
            never does it twice;
  display — shown to the user straight away (None to skip).

The turn waits for its reflexes, so a reflex must decide quickly that a message is not
for it.  One that takes longer than REFLEX_TIMEOUT_SECONDS is dropped, so a reflex that
acts on the world bounds its own deciding well inside that and leaves the rest for the
action itself: one cut off mid-action would leave no note of what it did.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# A reflex past this is dropped and the turn goes ahead without it.
REFLEX_TIMEOUT_SECONDS = 12.0


@dataclass(frozen=True)
class Exchange:
    """One earlier message in the conversation."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ReflexOutput:
    """What a reflex did."""

    note: str
    """Added to the user's message for the agent: what was already done."""

    display: str | None = None
    """Quick confirmation shown directly to the user.  None to skip."""


Reflex = Callable[[str, Sequence[Exchange]], Awaitable[ReflexOutput | None]]
"""
async def my_reflex(query: str, history: Sequence[Exchange]) -> ReflexOutput | None:
    ...

*history* is oldest first and does not include *query*.  Return None if the reflex has
nothing to do for this message.
"""


async def run_reflexes(
    reflexes: Sequence[Reflex],
    query: str,
    history: Sequence[Exchange],
    reflex_timeout: float = REFLEX_TIMEOUT_SECONDS,
) -> list[ReflexOutput]:
    """Run all reflexes in parallel and return the non-None results.

    Each reflex is given *reflex_timeout* seconds.  Timeouts and exceptions are logged and
    swallowed: a misbehaving reflex must never break the turn.
    """
    if not reflexes:
        return []

    async def _run_one(reflex: Reflex) -> ReflexOutput | None:
        name = getattr(reflex, "__name__", repr(reflex))
        try:
            result = await asyncio.wait_for(reflex(query, history), timeout=reflex_timeout)
        except TimeoutError:
            logger.warning("reflex_timeout reflex=%s timeout=%.1fs", name, reflex_timeout)
            return None
        except Exception:
            logger.exception("reflex_error reflex=%s", name)
            return None
        if result is None:
            logger.info("reflex_skip reflex=%s", name)
        else:
            logger.info("reflex_done reflex=%s note=%.120r", name, result.note)
        return result

    results = await asyncio.gather(*(_run_one(r) for r in reflexes))
    return [r for r in results if r is not None]
