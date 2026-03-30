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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from aug.config import get_settings
from aug.core.agents.base_agent import BaseAgent
from aug.core.events import AgentEvent, ChatModelStreamEvent
from aug.core.prompts import MID_RUN_INJECTION_PREFIX
from aug.core.reflexes import Reflex, ReflexOutput, run_reflexes
from aug.core.registry import get_agent
from aug.core.run import AGENT_RUN_CONFIG_KEY, AgentRun, MessageContent, run_registry
from aug.core.state import AgentState
from aug.utils.logging import set_correlation_id, set_thread_id
from aug.utils.notify import register_notification_target

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
    latitude: float
    longitude: float


ContentPart = TextContent | FileContent | LocationContent


class IncomingMessage(BaseModel):
    """Incoming message after translation from platform-specific input."""

    parts: list[ContentPart]
    interface: Literal["telegram", "rest_api"]
    sender_id: str
    thread_id: str
    agent_version: str


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseInterface[ContextT](ABC):
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        self._checkpointer = checkpointer

    @abstractmethod
    async def receive_message(self, context: ContextT) -> IncomingMessage | None:
        """Translate platform-specific input into a message.

        Return None to silently ignore the input (e.g. unauthorized sender).
        """

    @abstractmethod
    async def send_notification(self, target_id: str, text: str) -> None:
        """Send a proactive message to a user outside of a request/response cycle.

        Must raise on failure so callers (e.g. reminder loop) can retry.
        """

    async def send_stream(self, stream: AsyncIterator[AgentEvent], context: ContextT) -> None:
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

    async def run(self, context: ContextT) -> None:
        """Route: inject into active run or start a new one."""
        incoming = await self.receive_message(context)
        if incoming is None:
            return
        set_correlation_id(str(uuid4())[:8])

        # Preprocess before the lock — may be slow (Whisper, geocoding) but needed for both paths.
        content = await _preprocess(incoming.parts)

        async with run_registry.thread_lock(incoming.thread_id):
            existing = run_registry.get(incoming.thread_id)
            if existing and existing.active and not existing.user_requested_stop.is_set():
                existing.inject_message(content)
                return

            # No active run — start a new one, cancelling any stale one.
            if existing and existing.active:
                existing.user_requested_stop.set()
            run = AgentRun()
            run_registry.set(incoming.thread_id, run)

        await self._execute_run(run, incoming, content, context)

    async def _execute_run(
        self,
        run: AgentRun,
        incoming: IncomingMessage,
        content: MessageContent,
        context: ContextT,
        fire_reflexes: bool = True,
    ) -> None:
        """Execute the full agent pipeline for a newly started run."""
        set_thread_id(incoming.thread_id)
        register_notification_target(incoming.thread_id, incoming.interface, incoming.sender_id)

        agent = get_agent(incoming.agent_version)

        # Fire reflexes in parallel with the agent run.  The task injects results
        # into the run as soon as they're available so the agent can pick them up
        # at the next interrupt_after=["call_tools"] pause — rather than always
        # landing as a leftover follow-up run after the agent has finished.
        # Reflexes only fire for real user input — not for system-injected follow-up runs.
        reflex_task: asyncio.Task[list[ReflexOutput]] | None = None
        if agent.reflexes and fire_reflexes:
            query = (
                content
                if isinstance(content, str)
                else " ".join(b["text"] for b in content if b.get("type") == "text")
            )
            history = await _recent_history(agent, incoming.thread_id, self._checkpointer)
            reflex_task = asyncio.create_task(
                self._run_reflexes_and_inject(agent.reflexes, query, history, run, context)
            )

        stream = _agent_stream(content, incoming, self._checkpointer, run)
        t0 = time.monotonic()
        logger.info(
            "request_start thread=%s run=%s interface=%s agent=%s",
            incoming.thread_id,
            run.id,
            incoming.interface,
            incoming.agent_version,
        )
        try:
            await self.send_stream(stream, context)
        except psycopg.OperationalError:
            logger.exception("DB connection lost")
            await self.send_message(
                "Database connection lost. Please try again in a moment.", context
            )
        except RateLimitError:
            logger.warning("LLM rate limit hit")
            await self.send_message(
                "Context window is full. Use /clear to start a fresh conversation.", context
            )
        except GraphRecursionError:
            logger.warning("Agent hit recursion limit")
            await self.send_message(
                "The agent got stuck in a loop and was stopped. Try rephrasing your request.",
                context,
            )
        except Exception:
            logger.exception("Unhandled error in agent pipeline")
            await self.send_message("Sorry, something went wrong.", context)
        finally:
            run.active = False
            run_registry.pop(incoming.thread_id)
            logger.info(
                "request_end thread=%s run=%s duration=%.2fs",
                incoming.thread_id,
                run.id,
                time.monotonic() - t0,
            )

        # Ensure the reflex task has completed and its inject has been placed.
        # If the reflex finished mid-run the inject was already consumed there;
        # if it finished after, it now sits in the leftover queue below.
        if reflex_task is not None:
            await reflex_task

        # A message that arrived during the final streaming phase (no interrupt point available)
        # sits unconsumed on the queue. Start a new run for it now.
        async with run_registry.thread_lock(incoming.thread_id):
            try:
                leftover = run.pending_agent_injection.get_nowait()
            except asyncio.QueueEmpty:
                return
            new_run = AgentRun()
            run_registry.set(incoming.thread_id, new_run)

        logger.info("leftover_injection thread=%s new_run=%s", incoming.thread_id, new_run.id)
        await self._execute_run(new_run, incoming, leftover, context, fire_reflexes=False)

    def stop_run(self, thread_id: str) -> bool:
        """Stop the active run for a thread. Returns True if there was one."""
        run = run_registry.get(thread_id)
        if run and run.active:
            run.request_stop()
            return True
        return False

    async def _run_reflexes_and_inject(
        self,
        reflexes: list[Reflex],
        query: str,
        history: list[MessageContent],
        run: AgentRun,
        context: ContextT,
    ) -> list[ReflexOutput]:
        """Run reflexes and inject each result into the run immediately on completion.

        By injecting as soon as the reflex finishes (rather than after the agent
        finishes), the inject lands in the pending queue while the agent may still
        be running — making it available at the next interrupt_after=["call_tools"]
        pause point instead of always becoming a leftover follow-up run.
        """
        results = await run_reflexes(reflexes, query, history)
        for result in results:
            run.inject_message(result.inject)
            if result.display:
                try:
                    await self.send_message(result.display, context)
                except Exception:
                    logger.warning("reflex_display_failed", exc_info=True)
        return results


# ---------------------------------------------------------------------------
# Private: pipeline internals
# ---------------------------------------------------------------------------


async def _agent_stream(
    content: MessageContent,
    message: IncomingMessage,
    checkpointer: BaseCheckpointSaver,
    run: AgentRun,
) -> AsyncIterator[AgentEvent]:
    """Streaming generator that loops across LangGraph interrupt boundaries.

    On the first pass the graph receives an AgentState with the user's message.
    After each interrupt_after=["call_tools"] pause the graph is resumed with
    either a Command(update=…) carrying an injected HumanMessage, or None to
    simply continue.  The loop exits when the graph finishes or cancel is set.
    """
    agent = get_agent(message.agent_version)
    config: RunnableConfig = {
        "configurable": {
            "thread_id": message.thread_id,
            AGENT_RUN_CONFIG_KEY: run,
        }
    }

    graph_input: AgentState | Command | None = AgentState(
        messages=[HumanMessage(content=content)],
        thread_id=message.thread_id,
        interface=message.interface,
    )

    while True:
        async for event in agent.astream_events(graph_input, config, checkpointer):
            if run.user_requested_stop.is_set():
                return
            yield event

        if run.user_requested_stop.is_set():
            return

        # Determine whether the graph finished or paused at interrupt_after=["call_tools"].
        graph_state = await agent.aget_state(config, checkpointer)
        if not graph_state.next:
            return  # graph reached END naturally

        # Graph paused — check for a queued soft interrupt.
        try:
            injection = run.pending_agent_injection.get_nowait()
            logger.info("soft_interrupt thread=%s run=%s", message.thread_id, run.id)
            graph_input = Command(
                update={"messages": [HumanMessage(content=_frame_injection(injection))]}
            )
        except asyncio.QueueEmpty:
            graph_input = None  # resume from where the graph left off


async def _recent_history(
    agent: BaseAgent,
    thread_id: str,
    checkpointer: BaseCheckpointSaver,
    n: int = 3,
) -> list[MessageContent]:
    """Return the last *n* human/AI messages from the thread's checkpoint as labeled strings."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await agent.aget_state(config, checkpointer)
    except Exception:
        logger.warning("history_fetch_failed thread=%s", thread_id, exc_info=True)
        return []
    messages = (snapshot.values or {}).get("messages", [])
    filtered = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    result: list[MessageContent] = []
    for msg in filtered[-n:]:
        prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
        if isinstance(msg.content, str):
            result.append(f"{prefix}: {msg.content}")
        elif isinstance(msg.content, list):
            texts = [
                b["text"] for b in msg.content if isinstance(b, dict) and b.get("type") == "text"
            ]
            if texts:
                result.append(f"{prefix}: {' '.join(texts)}")
    return result


def _frame_injection(content: MessageContent) -> MessageContent:
    """Prepend MID_RUN_INJECTION_PREFIX so the LLM knows not to abandon its current task."""
    if isinstance(content, str):
        return MID_RUN_INJECTION_PREFIX + content
    # Multimodal: prepend a text block before the existing blocks.
    return [{"type": "text", "text": MID_RUN_INJECTION_PREFIX}, *content]


async def _collect(stream: AsyncIterator[AgentEvent]) -> str:
    text = ""
    async for event in stream:
        match event:
            case ChatModelStreamEvent(delta=delta) if delta:
                text += delta
    return text


# ---------------------------------------------------------------------------
# Private: preprocessing utilities
# ---------------------------------------------------------------------------


async def _preprocess(parts: list[ContentPart]) -> MessageContent:
    blocks = []
    for part in parts:
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, FileContent):
            if part.mime_type.startswith("image/"):
                blocks.append({"type": "text", "text": f"[[img:{part.path}|{part.mime_type}]]"})
            elif part.transcribe:
                transcript = await _transcribe(await part.read(), part.mime_type)
                blocks.append({"type": "text", "text": transcript})
                blocks.append({"type": "text", "text": f"[Audio saved to: {part.path}]"})
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            f"[File uploaded: {part.filename}"
                            f" ({part.mime_type}) — saved to {part.path}]"
                        ),
                    }
                )
        elif isinstance(part, LocationContent):
            text = await _geocode(part.latitude, part.longitude)
            blocks.append({"type": "text", "text": text})

    text_only = all(b["type"] == "text" for b in blocks)
    if text_only:
        return "\n\n".join(b["text"] for b in blocks)
    return blocks


async def _transcribe(data: bytes, mime_type: str) -> str:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.ogg", io.BytesIO(data), mime_type),
    )
    return result.text


async def _geocode(latitude: float, longitude: float) -> str:
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
    return f"User's current location:\nAddress: {display}\nCoordinates: {latitude}, {longitude}"
