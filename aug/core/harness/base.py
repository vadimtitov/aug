"""The Harness base class: one agent turn in, a stream of events out.

Every harness implements every abstract method; Python refuses to instantiate one that
doesn't, so a gap fails at boot rather than mid-conversation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import replace

from aug.core.harness.types import (
    ApprovalRequested,
    ApprovalResolved,
    ContextUsage,
    FileEmitted,
    FilePart,
    HarnessError,
    HarnessEvent,
    Message,
    ModelTier,
    SessionStatus,
    StatusUpdate,
    TextDelta,
    TextPart,
    ToolEnded,
    ToolStarted,
    TurnContext,
    TurnEnded,
    TurnStarted,
    UnsolicitedSink,
)
from aug.core.prompts import HARNESS_ERROR_MESSAGES, HARNESS_RETRY_AT_SUFFIX, REFLEX_NOTE
from aug.core.reflexes import Exchange, Reflex, ReflexOutput, run_reflexes
from aug.core.transcript import Transcript, file_record

logger = logging.getLogger(__name__)

# How long reading the context after a turn may hold the turn's end back.
CONTEXT_AFTER_TURN_SECONDS = 10
# Earlier messages a reflex is shown, for follow-ups like "now turn it off".
REFLEX_HISTORY = 4


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
        last is always TurnEnded, carrying any error and how full the context now is.
        The turn is logged and written to AUG's transcript here, so every harness
        records it the same way.  A tool whose approval was not granted ends with
        ``ToolEnded(error=True)``.

        The profile's reflexes run first, on a turn someone is there for (not a push,
        a job or a follow-up of leftover messages): what they did is added to the
        message the harness's agent sees, and their display text follows TurnStarted.
        """
        transcript = Transcript(turn.reply_to.conversation_id)
        reflexed = await self._run_reflexes(message, context_id, turn, transcript)
        seen = message
        if reflexed:
            notes = [TextPart(text=REFLEX_NOTE.format(note=r.note)) for r in reflexed]
            seen = replace(message, parts=[*message.parts, *notes])
        async for event in self._turn(
            message, context_id, turn, self._run(seen, context_id, turn), reflexed=reflexed
        ):
            yield event

    @property
    def reflexes(self) -> Sequence[Reflex]:
        """Run on each message before the turn (see run()).  Default: none."""
        return ()

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
    async def session_prompt(self, context_id: str, interface: str) -> str:
        """The system prompt the session runs with, or the one it will start with if it
        has not started yet."""

    @abstractmethod
    async def session_status(self, context_id: str) -> SessionStatus:
        """What the harness knows about the session: its model, how full its context is,
        the plan's usage limits, background work.  Never starts a session."""

    @abstractmethod
    async def oneshot(
        self, prompt: str, tier: ModelTier = ModelTier.STRONG, *, system: str = ""
    ) -> str:
        """A plain model call: no tools, no history.  Returns the text.  Used by jobs.

        *tier* picks the harness's model for the call; callers never name a model."""

    @asynccontextmanager
    async def running(self, on_unsolicited: UnsolicitedSink) -> AsyncIterator[None]:
        """The harness's whole life, entered once from the FastAPI lifespan.

        Default: nothing to start or stop.
        """
        yield

    async def _context_after_turn(self, context_id: str) -> ContextUsage | None:
        """The session's context once a turn whose TurnEnded didn't carry it is over.
        Default: session_status's.  Bounded by CONTEXT_AFTER_TURN_SECONDS."""
        return (await self.session_status(context_id)).context

    async def _safe_context(self, context_id: str) -> ContextUsage | None:
        """_context_after_turn, never failing or stalling the turn: None instead."""
        try:
            return await asyncio.wait_for(
                self._context_after_turn(context_id), CONTEXT_AFTER_TURN_SECONDS
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # TimeoutError included
            logger.warning("context after turn failed context=%s: %r", context_id, e)
            return None

    async def _run_reflexes(
        self, message: Message, context_id: str, turn: TurnContext, transcript: Transcript
    ) -> list[ReflexOutput]:
        if not self.reflexes or not turn.attended or turn.follow_up or not message.text:
            return []
        records = await transcript.recent(context_id, REFLEX_HISTORY)
        history = [
            Exchange(role="user" if r["event"] == "turn_started" else "assistant", text=r["text"])
            for r in records
        ]
        return await run_reflexes(self.reflexes, message.text, history)

    def _open_turn(self, context_id: str, turn: TurnContext) -> None:  # noqa: B027 — optional hook
        """Called before TurnStarted: from here on inject() must reach this turn."""

    def _close_turn(self, context_id: str) -> None:  # noqa: B027 — optional hook
        """Called once the turn is over, however it ended."""

    async def _turn(
        self,
        message: Message,
        context_id: str,
        turn: TurnContext,
        inner: AsyncIterator[HarnessEvent],
        *,
        reflexed: Sequence[ReflexOutput] = (),
    ) -> AsyncIterator[HarnessEvent]:
        """Wrap a harness's own event stream in the run() contract.  *reflexed*: what
        the reflexes did before the turn, shown and recorded once it starts."""
        transcript = Transcript(turn.reply_to.conversation_id)
        started = time.monotonic()
        tool_started: dict[str, float] = {}
        not_granted: set[str] = set()  # tool ids whose approval was not granted
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
                user_id=message.sender.user_id,
                text=message.text,
                files=[
                    await file_record(p.path, p.media_type, p.filename)
                    for p in message.parts
                    if isinstance(p, FilePart)
                ],
            )
            if reflexed:
                await transcript.append("reflexed", context_id, notes=[r.note for r in reflexed])
            for r in reflexed:
                if r.display:
                    yield StatusUpdate(text=r.display)
            try:
                async with aclosing(inner) as events:
                    async for event in events:
                        match event:
                            case TurnEnded():
                                ended = event
                                break
                            case TextDelta(text=delta, parent_id=None):
                                text += delta
                            case ToolStarted(id=tool_id, name=name, args=args):
                                tool_started[tool_id] = time.monotonic()
                                logger.info("tool_start tool=%s args=%.120r", name, args)
                                await transcript.append(
                                    "tool_started", context_id, tool=name, args=args
                                )
                            case ApprovalRequested(request=request):
                                await transcript.append(
                                    "approval_requested",
                                    context_id,
                                    tool=request.tool_name,
                                    operation=request.description,
                                )
                            case ApprovalResolved(decision=decision, parent_id=parent_id):
                                if not decision.approved and parent_id is not None:
                                    not_granted.add(parent_id)
                                await transcript.append(
                                    "approval_resolved", context_id, decision=decision.value
                                )
                            case FileEmitted(path=path, media_type=media_type, filename=name):
                                await transcript.append(
                                    "file_sent",
                                    context_id,
                                    **await file_record(path, media_type, name),
                                )
                            case ToolEnded(id=tool_id, name=name) if tool_id in not_granted:
                                event = replace(event, error=True)
                                _log_tool_end(name, True, tool_started.pop(tool_id, None))
                            case ToolEnded(id=tool_id, name=name, error=error):
                                _log_tool_end(name, error, tool_started.pop(tool_id, None))
                        yield event
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("turn crashed context=%s: %s: %s", context_id, type(e).__name__, e)
                ended = TurnEnded(error=HarnessError(kind="internal"))

            ended = ended or TurnEnded()
            if ended.context is None:
                ended = replace(ended, context=await self._safe_context(context_id))
            await transcript.append(
                "turn_ended",
                context_id,
                text=text,
                error=ended.error.kind if ended.error else None,
                cancelled=ended.cancelled,
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
