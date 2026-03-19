"""Base interface protocol and shared types for all AUG frontends.

Each frontend (Telegram, REST API, etc.) subclasses BaseInterface[ContextT] and implements:
  - receive_message: translate platform input into IncomingMessage
  - send_stream: consume the agent event stream and deliver progressively (optional)
  - send_message: deliver a complete response in one shot

The base class owns the full pipeline: preprocess → agent → deliver.
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from openai import AsyncOpenAI
from pydantic import BaseModel

from aug.config import get_settings
from aug.core.events import AgentEvent, ChatModelStreamEvent
from aug.core.registry import get_agent
from aug.core.state import AgentState
from aug.utils.logging import set_correlation_id
from aug.utils.notify import register_notification_target

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    text: str


class AudioContent(BaseModel):
    data: bytes
    mime_type: str = "audio/ogg"


class ImageContent(BaseModel):
    data: bytes
    mime_type: str = "image/jpeg"


class LocationContent(BaseModel):
    latitude: float
    longitude: float


ContentPart = TextContent | AudioContent | ImageContent | LocationContent


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
        """Full pipeline: receive → preprocess → agent → deliver."""
        set_correlation_id(str(uuid4())[:8])
        incoming = await self.receive_message(context)
        if incoming is None:
            return
        register_notification_target(incoming.thread_id, incoming.interface, incoming.sender_id)
        content = await _preprocess(incoming.parts)
        stream = _stream_agent(content, incoming, self._checkpointer)
        await self.send_stream(stream, context)


# ---------------------------------------------------------------------------
# Private: pipeline internals
# ---------------------------------------------------------------------------


async def _preprocess(parts: list[ContentPart]) -> str | list:
    blocks = []
    for part in parts:
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, AudioContent):
            transcript = await _transcribe(part.data, part.mime_type)
            blocks.append({"type": "text", "text": transcript})
        elif isinstance(part, ImageContent):
            encoded = base64.b64encode(part.data).decode()
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.mime_type};base64,{encoded}"},
                }
            )
        elif isinstance(part, LocationContent):
            text = await _geocode(part.latitude, part.longitude)
            blocks.append({"type": "text", "text": text})

    text_only = all(b["type"] == "text" for b in blocks)
    if text_only:
        return "\n\n".join(b["text"] for b in blocks)
    return blocks


def _stream_agent(
    content: str | list,
    message: IncomingMessage,
    checkpointer: BaseCheckpointSaver,
) -> AsyncIterator[AgentEvent]:
    agent = get_agent(message.agent_version)
    state = AgentState(
        messages=[HumanMessage(content=content)],
        thread_id=message.thread_id,
        interface=message.interface,
    )
    config: RunnableConfig = {"configurable": {"thread_id": message.thread_id}}
    return agent.astream_events(state, config, checkpointer)


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
