"""Base interface protocol and shared types for all AUG frontends.

Each frontend (Telegram, REST API, etc.) subclasses BaseInterface[ContextT] and implements:
  - receive_message: translate platform input into IncomingMessage
  - send_stream: consume the agent event stream and deliver progressively (optional)
  - send_message: deliver a complete response in one shot

The base class owns the full pipeline: preprocess → agent → deliver.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from aug.config import get_settings
from aug.core.events import ChatModelStreamEvent
from aug.core.harness import (
    Address,
    FilePart,
    Harness,
    HarnessError,
    HarnessEvent,
    Message,
    Sender,
    TextPart,
    TurnContext,
    TurnEnded,
    error_text,
)
from aug.core.harness.langgraph import LangGraphHarness
from aug.core.prompts import LOCATION_SHARE_TEMPLATE, LOCATION_SHARE_UNKNOWN_SENDER
from aug.core.registry import get_harness
from aug.core.run import run_registry
from aug.core.tools.approval import ApprovalDecision, ApprovalRequest
from aug.utils.file_settings import ConversationSettings, load_settings, save_settings
from aug.utils.logging import set_correlation_id, set_thread_id
from aug.utils.state import ConversationLocationState, LiveLocationState, load_state, save_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    text: str


class FileContent(BaseModel):
    """A file received from any interface, persisted to disk.

    Attributes:
        path:       Absolute path where the file is stored on disk.
        mime_type:  MIME type of the file (e.g. ``"image/jpeg"``).
        transcribe: If True, ``_preprocess`` will transcribe the audio to text
                    (intended for voice notes, not music/general audio files).
    """

    path: str
    mime_type: str
    transcribe: bool = False

    @property
    def filename(self) -> str:
        return Path(self.path).name

    async def write(self, data: bytes) -> None:
        """Write *data* to ``path``, creating parent directories as needed."""
        await asyncio.to_thread(self._write_sync, data)

    def _write_sync(self, data: bytes) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_bytes(data)

    async def read(self) -> bytes:
        """Read file bytes from disk.

        Raises:
            FileNotFoundError: if the file is not on disk.
        """
        return await asyncio.to_thread(Path(self.path).read_bytes)


class LocationContent(BaseModel):
    """A position shared from any interface.

    Attributes:
        live_period: Seconds the sharer granted for live updates, if this is a live
                     share.  None for a one-off pin, and for a live update that does
                     not restate the period.
        sent_at:     Unix time the message was sent, which is what ``live_period``
                     counts from.  An edit of a live share keeps the original send
                     date, so this anchors the expiry instead of letting each update
                     push it further out.
        reported_at: Unix time the platform stamped on this particular position (an
                     edit date, where there is one).  Orders updates against each
                     other so a late delivery cannot replace a newer position.
        sender_name: Display name of the person sharing, for the text the agent reads.
                     Without it two group members' positions are indistinguishable.

    Both timestamps are None on interfaces that do not supply them; the receipt time
    is used instead.
    """

    latitude: float
    longitude: float
    live_period: int | None = None
    sent_at: float | None = None
    reported_at: float | None = None
    sender_name: str | None = None


ContentPart = TextContent | FileContent | LocationContent


class IncomingMessage(BaseModel):
    """Incoming message after translation from platform-specific input."""

    parts: list[ContentPart]
    interface: Literal["telegram", "rest_api"]
    sender_id: str  # delivery target — the chat a reply goes back to
    thread_id: str
    agent_version: str
    # Platform id of the person who sent this.  Distinct from sender_id, which addresses
    # a place: in a group chat many user_ids share one sender_id.
    user_id: str = ""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseInterface[ContextT](ABC):
    # Subclasses may override to buffer rapid-fire messages before starting a run.
    # 0.0 = disabled (start immediately).
    _debounce_window: float = 0.0

    def __init__(self) -> None:
        self._debounce_buf: dict[str, list[tuple[IncomingMessage, Message]]] = {}
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}

    @abstractmethod
    async def receive_message(self, context: ContextT) -> IncomingMessage | None:
        """Translate platform-specific input into a message.

        Return None to silently ignore the input (e.g. unauthorized sender).
        """

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest, context: ContextT) -> None:
        """Send an approval prompt to the user (fire-and-forget).

        Must deliver the prompt and return immediately — do NOT await the user's
        response here.  The run will exit after this call; resumption happens via
        a separate entry point (e.g. Telegram callback handler, REST endpoint)
        that calls _execute_resume with the user's ApprovalDecision.
        """

    @abstractmethod
    async def resolve_thread(
        self,
        thread_id: str,
        *,
        topic_name: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """Resolve a logical thread_id to a concrete thread ID for this interface.

        Accepts:
          ``"default"``    — the interface's canonical default delivery target
                            (e.g. the DM chat for Telegram).
          ``"new"``        — create a new thread (e.g. a Telegram forum topic).
                            Requires ``chat_id`` for group-based interfaces and
                            accepts an optional ``topic_name``.
          any other value  — returned as-is (assumed to be a valid thread ID).

        Raises:
            ValueError: if the thread cannot be resolved (e.g. no default chat).
        """

    @abstractmethod
    def conversation_id(self, thread_id: str) -> str:
        """Return the stable conversation key that *thread_id* belongs to.

        Thread IDs may rotate — Telegram's /clear starts a new session and thus a
        new thread ID — but per-conversation settings must survive that.  The
        conversation key is the identity of the place the user is talking in
        (a DM, a forum topic, a REST thread), and must be unique across
        interfaces so keys from different frontends never collide.
        """

    @abstractmethod
    async def send_proactive(self, thread_id: str, text: str) -> None:
        """Send a plain-text message to *thread_id* without an agent turn.

        Used for ``type="forward"`` pushes.  Must raise on failure.
        """

    @abstractmethod
    async def announcement_threads(self) -> list[str]:
        """Return the threads that should receive service-level announcements.

        An announcement belongs to the service rather than to a conversation — a
        startup notice, for example — so the core broadcasts it without naming a
        frontend.  Each interface answers for itself: none (no push channel at
        all), one, or one per owner.
        """

    @abstractmethod
    async def send_proactive_stream(
        self, thread_id: str, stream: AsyncIterator[HarnessEvent]
    ) -> None:
        """Consume an agent event stream and deliver the final response to *thread_id*.

        Only the accumulated final text and any tool output attachments are
        sent — intermediate streaming updates and tool call messages are
        suppressed.  Used for ``type="agent"`` pushes and scheduled tasks.
        """

    async def send_stream(self, stream: AsyncIterator[HarnessEvent], context: ContextT) -> None:
        """Consume the agent event stream and deliver the response.

        Default: collect the full response then call send_message.
        Override for platforms that support live streaming (e.g. Telegram).
        """
        full = await _collect(stream)
        await self.send_message(full, context)

    async def send_message(self, message: str, context: ContextT) -> None:
        """Deliver a complete response in one shot.

        Must be implemented if send_stream is not overridden.
        """
        raise NotImplementedError

    def parent_conversation_id(self, conversation_id: str) -> str | None:
        """Return the broader conversation *conversation_id* inherits settings from.

        Default: none — every conversation stands alone.  Telegram overrides this so a
        forum topic falls back to its group's selection until one is picked in the topic
        itself.  Without a fallback, a conversation nobody has visited yet resolves to no
        agent at all, which is fine for a chat (the user is told to run /version) but not
        for a scheduled push, where there is nobody to tell.
        """
        return None

    def get_agent_version(self, thread_id: str) -> str:
        """Return the agent version selected for *thread_id*'s conversation.

        Falls back to the parent conversation when this one has no selection of its own.
        """
        conversations = load_settings().conversations
        conversation = self.conversation_id(thread_id)
        if conversation in conversations:
            return conversations[conversation].agent
        parent = self.parent_conversation_id(conversation)
        if parent is not None and parent in conversations:
            return conversations[parent].agent
        return ConversationSettings().agent

    def set_agent_version(self, thread_id: str, agent: str) -> None:
        """Select *agent* for *thread_id*'s conversation, leaving other conversations alone."""
        conversation = self.conversation_id(thread_id)
        settings = load_settings()
        settings.conversations.setdefault(conversation, ConversationSettings()).agent = agent
        save_settings(settings)

    def live_locations(self, thread_id: str) -> dict[str, LiveLocationState]:
        """Latest location per user for *thread_id*'s conversation, keyed by user id.

        Includes users whose sharing period has already expired — callers decide what
        counts as stale via ``LiveLocationState.is_live`` / ``age_seconds``.
        """
        return self._location_state(thread_id).users

    def record_location(self, thread_id: str, user_id: str, location: LocationContent) -> bool:
        """Persist *location* as *user_id*'s latest position in *thread_id*'s conversation.

        Called for every location that arrives, whoever sent it, so several people in
        one group are tracked side by side.  An update the platform timestamped before
        the one already stored is dropped: delivery reorders, and a stale position must
        not displace a fresh one.

        Returns False when the position was dropped as stale, so the caller can leave
        the agent out of it too — showing it coordinates that are known to be outdated
        is worse than saying nothing.
        """
        now = time.time()
        state = load_state()
        conversation = state.locations.setdefault(
            self.conversation_id(thread_id), ConversationLocationState()
        )
        previous = conversation.users.get(user_id, LiveLocationState(user_id=user_id))
        if location.reported_at and location.reported_at < previous.reported_at:
            logger.debug("dropping out-of-order location for %s in %s", user_id, thread_id)
            return False

        conversation.users[user_id] = LiveLocationState(
            user_id=user_id,
            latitude=location.latitude,
            longitude=location.longitude,
            updated_at=now,
            reported_at=location.reported_at or 0.0,
            live_until=_live_until(location, previous, now),
        )
        save_state(state)
        return True

    def claim_location_run(self, thread_id: str) -> bool:
        """Reserve a location-triggered agent run for *thread_id*'s conversation.

        Returns True and stamps the throttle if enough time has passed since the last
        such run, False if the update should be recorded but not woken on.  Only for
        runs the location itself triggers — injecting into a run that is already going
        is free and should not consume the throttle.
        """
        now = time.time()
        state = load_state()
        conversation = state.locations.setdefault(
            self.conversation_id(thread_id), ConversationLocationState()
        )
        if now - conversation.last_run_at < conversation.throttle_seconds:
            return False
        conversation.last_run_at = now
        save_state(state)
        return True

    def stamp_location_run(self, thread_id: str) -> None:
        """Record that a location is starting an agent run now, throttle or no throttle.

        The decision to run has already been made by the time this is called — a first
        share, or an update that won the race against a run finishing — so the clock
        still has to be reset, or the very next update wakes the agent again.
        """
        state = load_state()
        state.locations.setdefault(
            self.conversation_id(thread_id), ConversationLocationState()
        ).last_run_at = time.time()
        save_state(state)

    def location_throttle(self, thread_id: str) -> int:
        """Minimum seconds between location-triggered agent runs in this conversation."""
        return self._location_state(thread_id).throttle_seconds

    def set_location_throttle(self, thread_id: str, seconds: int) -> None:
        """Set the minimum seconds between location-triggered runs in this conversation."""
        state = load_state()
        state.locations.setdefault(
            self.conversation_id(thread_id), ConversationLocationState()
        ).throttle_seconds = seconds
        save_state(state)

    async def run(self, context: ContextT) -> None:
        """Route: inject into the turn in flight, or start a new one."""
        incoming = await self.receive_message(context)
        if incoming is None:
            return
        set_correlation_id(str(uuid4())[:8])

        # Record before preprocessing: the very first share must land in the state too,
        # not only the live updates that follow it, or a reader sees an empty gap.
        # Fall back to the sender when an interface has no per-user id, so locations
        # from different places cannot pile up under one empty key.
        located = False
        for part in incoming.parts:
            if isinstance(part, LocationContent):
                accepted = self.record_location(
                    incoming.thread_id, incoming.user_id or incoming.sender_id, part
                )
                located = located or accepted

        # Preprocess before the lock — may be slow (Whisper, geocoding).
        message = await _preprocess(incoming)

        fire_inline = False
        async with run_registry.thread_lock(incoming.thread_id):
            harness = _harness_for(incoming.agent_version)
            if harness is not None and await harness.inject(message, incoming.thread_id):
                return

            # A location is about to start a run.  Stamp the throttle here, under the
            # lock, because this is where the decision is actually taken: the caller's
            # check happened before preprocessing, and the run it saw may since have
            # finished.  Covers the first share too, which never passes such a check.
            if located:
                self.stamp_location_run(incoming.thread_id)

            # No turn in flight — buffer for the debounce window (or fire at once).
            self._debounce_buf.setdefault(incoming.thread_id, []).append((incoming, message))
            task = self._debounce_tasks.get(incoming.thread_id)
            if task is None or task.done():
                if self._debounce_window == 0:
                    fire_inline = True  # await _debounce_fire after releasing the lock
                else:
                    self._debounce_tasks[incoming.thread_id] = asyncio.create_task(
                        self._debounce_fire(incoming.thread_id, context)
                    )

        if fire_inline:
            await self._debounce_fire(incoming.thread_id, context)

    async def _debounce_fire(self, thread_id: str, context: ContextT) -> None:
        """Wait out the debounce window, then start a single merged turn."""
        if self._debounce_window > 0:
            await asyncio.sleep(self._debounce_window)

        entries = self._debounce_buf.pop(thread_id, [])
        self._debounce_tasks.pop(thread_id, None)
        if not entries:
            return
        await self._converse(entries[0][0], _merge_messages([m for _, m in entries]), context)

    async def _converse(
        self, incoming: IncomingMessage, message: Message, context: ContextT
    ) -> None:
        """Run *message* as a turn, then a follow-up turn for anything injected too late."""
        set_thread_id(incoming.thread_id)
        harness = _harness_for(incoming.agent_version)
        if harness is None:
            await self.send_message(
                "No agent selected. Use /version to pick one before chatting.", context
            )
            return

        turn = self._turn_context(incoming.interface, incoming.thread_id, incoming.sender_id)
        async with run_registry.run_lock(incoming.thread_id):
            next_message: Message | None = message
            while next_message is not None:
                ended = await self._deliver(
                    lambda message=next_message: harness.run(message, incoming.thread_id, turn),
                    incoming.thread_id,
                    context,
                )
                next_message = None
                if ended and ended.unconsumed and not ended.error and not ended.approval:
                    # Arrived during the final streaming phase, with no tool-round pause
                    # left to take it: it gets a turn of its own, straight away.
                    logger.info("leftover_injection thread=%s", incoming.thread_id)
                    next_message = _merge_messages(ended.unconsumed)

    async def _deliver(
        self,
        start: Callable[[], AsyncGenerator[HarnessEvent]],
        thread_id: str,
        context: ContextT,
    ) -> TurnEnded | None:
        """Stream one turn to the user, then report its error or ask its approval.

        The turn is started under the routing lock and held there until TurnStarted,
        so a message arriving in between is injected rather than queued behind it.
        Returns the TurnEnded, or None when delivering the stream itself failed.
        """
        async with run_registry.thread_lock(thread_id):
            stream = start()
            first = await anext(stream)
        ended: TurnEnded | None = None

        async def events() -> AsyncIterator[HarnessEvent]:
            nonlocal ended
            yield first
            async for event in stream:
                if isinstance(event, TurnEnded):
                    ended = event
                yield event

        async with aclosing(stream):
            try:
                await self.send_stream(events(), context)
                if ended is not None and ended.error is None and ended.approval is not None:
                    await self.request_approval(ended.approval, context)
            except Exception as e:
                logger.exception("turn delivery failed: %s: %s", type(e).__name__, e)
                await self._send_quietly(error_text(HarnessError(kind="internal")), context)
                return None
        if ended is not None and ended.error is not None:
            await self._send_quietly(error_text(ended.error), context)
        return ended

    async def _send_quietly(self, text: str, context: ContextT) -> None:
        try:
            await self.send_message(text, context)
        except Exception:
            logger.warning("failed to deliver error message to user")

    def _turn_context(self, interface: str, thread_id: str, chat_id: str) -> TurnContext:
        return TurnContext(
            reply_to=Address(
                interface=interface,
                conversation_id=self.conversation_id(thread_id),
                chat_id=chat_id,
            ),
            attended=True,
        )

    def _location_state(self, thread_id: str) -> ConversationLocationState:
        """Stored location state for *thread_id*'s conversation, defaults if never used."""
        return load_state().locations.get(
            self.conversation_id(thread_id), ConversationLocationState()
        )

    async def stop_run(self, thread_id: str) -> bool:
        """Stop the turn in flight for a thread.  Returns True if there was one."""
        harness = _harness_for(self.get_agent_version(thread_id))
        return harness is not None and await harness.cancel(thread_id)

    async def get_pending_approval(
        self, thread_id: str, agent_version: str
    ) -> ApprovalRequest | None:
        """Return the pending ApprovalRequest for *thread_id*, or None if not paused."""
        harness = _harness_for(agent_version)
        if not isinstance(harness, LangGraphHarness):
            return None
        return await harness.pending_approval(thread_id)

    async def _execute_compact(self, thread_id: str, agent_version: str) -> bool:
        """Compact the message history for *thread_id*, waiting for any turn in flight.

        Returns True if compaction ran, False if there was nothing to compact.
        Raises ValueError if the agent has no compaction_model configured.
        """
        harness = get_harness(agent_version)
        async with run_registry.run_lock(thread_id):
            return await harness.compact(thread_id)

    async def _execute_resume(
        self,
        thread_id: str,
        agent_version: str,
        sender_id: str,
        interface: Literal["telegram", "rest_api"],
        decision: ApprovalDecision,
        context: ContextT,
    ) -> None:
        """Resume a turn paused on an approval with the user's *decision*.

        Called by each interface's approval-response handler (Telegram button
        callback, REST /chat/approve endpoint) after the user has responded.
        """
        harness = _harness_for(agent_version)
        if not isinstance(harness, LangGraphHarness):
            logger.warning("approval_resume: %r has no resumable approvals", agent_version)
            return
        set_thread_id(thread_id)
        turn = self._turn_context(interface, thread_id, sender_id)
        async with run_registry.run_lock(thread_id):
            logger.info("approval_resume thread=%s decision=%s", thread_id, decision.value)
            await self._deliver(
                lambda: harness.resume(decision, thread_id, turn), thread_id, context
            )


# ---------------------------------------------------------------------------
# Private: pipeline internals
# ---------------------------------------------------------------------------


def _harness_for(agent_version: str) -> Harness | None:
    """The harness for *agent_version*, or None when no such version is registered."""
    try:
        return get_harness(agent_version)
    except ValueError:
        return None


def _merge_messages(messages: list[Message]) -> Message:
    """Merge messages that arrived together into one, keeping the first sender."""
    if len(messages) == 1:
        return messages[0]
    parts = [part for m in messages for part in m.parts]
    return Message(parts=parts, sender=messages[0].sender)


async def _collect(stream: AsyncIterator[HarnessEvent]) -> str:
    text = ""
    async for event in stream:
        match event:
            case ChatModelStreamEvent(delta=delta) if delta:
                text += delta
    return text


# ---------------------------------------------------------------------------
# Private: preprocessing utilities
# ---------------------------------------------------------------------------


def _live_until(location: LocationContent, previous: LiveLocationState, now: float) -> float:
    """Unix time the live sharing period ends, or 0.0 if this position is not live.

    ``live_period`` counts from when the message was sent, not from when it reached us,
    so the expiry is anchored to the send date.  Recomputing it for each edit of the
    same share therefore lands on the same answer: an update cannot push the window
    further out, and one delivered after the window closed cannot revive it.

    A platform sends ``live_period`` with an active live share.  Without it the update
    is either an edit that simply omits the field — the known expiry stands — or a
    one-off pin, which clears whatever live status an earlier share left behind.
    """
    if location.live_period:
        return (location.sent_at or now) + location.live_period
    return previous.live_until if previous.is_live(now) else 0.0


async def _preprocess(incoming: IncomingMessage) -> Message:
    """Turn what an interface received into the harness Message: files stay files,
    voice notes become their transcript, locations become an address."""
    parts: list[TextPart | FilePart] = []
    for part in incoming.parts:
        if isinstance(part, TextContent):
            parts.append(TextPart(text=part.text))
        elif isinstance(part, FileContent):
            if part.transcribe:
                transcript = await _transcribe(await part.read(), part.mime_type)
                parts.append(
                    TextPart(
                        text=(
                            f"[Voice message, already transcribed]\n"
                            f"{transcript}\n"
                            f"[Audio saved to: {part.path}]"
                        )
                    )
                )
            else:
                parts.append(
                    FilePart(
                        path=Path(part.path), media_type=part.mime_type, filename=part.filename
                    )
                )
        elif isinstance(part, LocationContent):
            text = await _geocode(part.latitude, part.longitude, part.sender_name)
            parts.append(TextPart(text=text))
    return Message(
        parts=parts, sender=Sender(interface=incoming.interface, user_id=incoming.user_id)
    )


async def _transcribe(data: bytes, mime_type: str) -> str:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.ogg", io.BytesIO(data), mime_type),
    )
    return result.text


async def _geocode(latitude: float, longitude: float, sender_name: str | None = None) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": latitude, "lon": longitude, "format": "json"},
            headers={"User-Agent": "aug-assistant/1.0"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    display = data.get("display_name", f"{latitude}, {longitude}")
    return LOCATION_SHARE_TEMPLATE.format(
        sender=sender_name or LOCATION_SHARE_UNKNOWN_SENDER,
        address=display,
        latitude=latitude,
        longitude=longitude,
    )
