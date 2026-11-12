"""The Harness base class: one agent turn in, a stream of events out.

Every harness implements every abstract method; Python refuses to instantiate one that
doesn't, so a gap fails at boot rather than mid-conversation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

from aug.core.events import ChatModelStreamEvent, ToolEndEvent, ToolStartEvent
from aug.core.harness.types import (
    FilePart,
    HarnessError,
    HarnessEvent,
    Message,
    ModelTier,
    TurnContext,
    TurnEnded,
    TurnStarted,
    UnsolicitedSink,
)
from aug.core.prompts import HARNESS_ERROR_MESSAGES, HARNESS_RETRY_AT_SUFFIX
from aug.core.transcript import Transcript

logger = logging.getLogger(__name__)


class Harness(ABC):
    """Runs agent turns, keyed by context_id (a chat's current session).

    The caller makes sure two turns never share a context_id at once by holding
    ``run_registry.run_lock(context_id)`` for the whole turn.
    """

    async def run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        """One agent turn.  Never raises.

        The first event is TurnStarted, yielded once the turn can take inject(); the
        last is always TurnEnded, carrying any error.  The turn is logged and written
        to AUG's transcript here, so every harness records it the same way.
        """
        async for event in self._turn(
            message, context_id, turn, self._run(message, context_id, turn)
        ):
            yield event

    @abstractmethod
    def _run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        """The harness's own turn.  May raise; may end with its own TurnEnded."""

    @abstractmethod
    async def inject(self, message: Message, context_id: str) -> bool:
        """Add *message* to the turn in flight.  False if nothing is running."""

    @abstractmethod
    async def cancel(self, context_id: str) -> bool:
        """Stop the turn and its running tool; it ends with TurnEnded(cancelled=True).
        False if nothing is running."""

    @abstractmethod
    async def compact(self, context_id: str) -> bool:
        """Summarise the session's history.  False if there was nothing to compact.
        The caller holds the run lock."""

    @abstractmethod
    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG) -> str:
        """A plain model call: no tools, no history.  Returns the text.  Used by jobs."""

    @asynccontextmanager
    async def running(self, on_unsolicited: UnsolicitedSink) -> AsyncIterator[None]:
        """The harness's whole life, entered once from the FastAPI lifespan.

        Default: nothing to start or stop.
        """
        yield

    def _open_turn(self, context_id: str, turn: TurnContext) -> None:  # noqa: B027 — optional hook
        """Called before TurnStarted: from here on inject() must reach this turn."""

    def _close_turn(self, context_id: str) -> None:  # noqa: B027 — optional hook
        """Called once the turn is over, however it ended."""

    async def _turn(
        self,
        message: Message | None,
        context_id: str,
        turn: TurnContext,
        inner: AsyncIterator[HarnessEvent],
    ) -> AsyncIterator[HarnessEvent]:
        """Wrap a harness's own event stream in the run() contract.

        Shared by run() and by harness-specific entry points that also stream a turn
        (LangGraph's approval resume), so all of them log and end the same way.
        """
        transcript = Transcript(turn.reply_to.conversation_id)
        started = time.monotonic()
        tool_started: dict[str, float] = {}
        text = ""
        ended: TurnEnded | None = None
        self._open_turn(context_id, turn)
        try:
            yield TurnStarted(context_id=context_id)
            logger.info(
                "turn_start context=%s interface=%s attended=%s",
                context_id,
                turn.reply_to.interface,
                turn.attended,
            )
            await transcript.append(
                "turn_started",
                context_id,
                interface=turn.reply_to.interface,
                attended=turn.attended,
                user_id=message.sender.user_id if message else "",
                text=message.text if message else "",
                files=[str(p.path) for p in message.parts if isinstance(p, FilePart)]
                if message
                else [],
            )
            try:
                async with aclosing(inner) as events:
                    async for event in events:
                        match event:
                            case TurnEnded():
                                ended = event
                                break
                            case ChatModelStreamEvent(delta=delta):
                                text += delta
                            case ToolStartEvent(run_id=run_id, tool_name=name, args=args):
                                tool_started[run_id] = time.monotonic()
                                logger.info("tool_start tool=%s args=%.120r", name, args)
                                await transcript.append(
                                    "tool_started", context_id, tool=name, args=args
                                )
                            case ToolEndEvent(run_id=run_id, tool_name=name, error=error):
                                _log_tool_end(name, error, tool_started.pop(run_id, None))
                        yield event
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("turn crashed context=%s: %s: %s", context_id, type(e).__name__, e)
                ended = TurnEnded(error=HarnessError(kind="internal"))

            ended = ended or TurnEnded()
            await transcript.append(
                "turn_ended",
                context_id,
                text=text,
                error=ended.error.kind if ended.error else None,
                cancelled=ended.cancelled,
                awaiting_approval=ended.approval is not None,
                unconsumed=len(ended.unconsumed),
            )
            logger.info(
                "turn_end context=%s duration=%.2fs error=%s cancelled=%s",
                context_id,
                time.monotonic() - started,
                ended.error.kind if ended.error else None,
                ended.cancelled,
            )
            yield ended
        finally:
            self._close_turn(context_id)


def error_text(error: HarnessError) -> str:
    """What the user is told about *error*."""
    text = HARNESS_ERROR_MESSAGES[error.kind]
    if error.retry_at is not None:
        text += HARNESS_RETRY_AT_SUFFIX.format(time=error.retry_at.strftime("%H:%M"))
    return text


def _log_tool_end(name: str, error: bool, started: float | None) -> None:
    elapsed = time.monotonic() - started if started is not None else 0.0
    if error:
        logger.warning("tool_end tool=%s duration=%.2fs error=True", name, elapsed)
    else:
        logger.info("tool_end tool=%s duration=%.2fs", name, elapsed)
