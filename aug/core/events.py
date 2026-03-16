"""Typed agent stream events.

LangChain's astream_events yields raw TypedDict objects with untyped `data`
fields. This module defines typed dataclasses for the events we actually handle
and a parse_event() function that converts raw StreamEvents into them.

Unrecognised events are returned as None so callers can skip them cleanly.

Each dispatchable event (currently ToolProgressEvent) also owns its own
dispatch() method so the event name and payload schema are defined exactly once.

Usage (receiving):
    async for raw in stream:
        match parse_event(raw):
            case ChatModelStreamEvent(delta=delta):
                ...
            case ToolProgressEvent(step=step):
                ...

Usage (dispatching from a tool):
    await ToolProgressEvent(step="Step 1 · example.com").dispatch()
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.schema import StreamEvent

from aug.core.tools.output import ToolOutput

_TOOL_PROGRESS = "tool_progress"


@dataclass
class ChatModelStreamEvent:
    delta: str


@dataclass
class ToolStartEvent:
    run_id: str
    tool_name: str
    args: dict


@dataclass
class ToolEndEvent:
    run_id: str
    tool_name: str
    output: ToolOutput | str | None


@dataclass
class ToolProgressEvent:
    step: str
    parent_ids: list[str] = field(default_factory=list)


async def send_tool_progress_update(step: str) -> None:
    await adispatch_custom_event(_TOOL_PROGRESS, {"step": step})


AgentEvent = ChatModelStreamEvent | ToolStartEvent | ToolEndEvent | ToolProgressEvent


def parse_event(event: StreamEvent) -> AgentEvent | None:
    kind = event["event"]
    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        delta = chunk.content if isinstance(chunk.content, str) else ""
        return ChatModelStreamEvent(delta=delta)
    if kind == "on_tool_start":
        return ToolStartEvent(
            run_id=event["run_id"],
            tool_name=event["name"],
            args=event["data"].get("input") or {},
        )
    if kind == "on_tool_end":
        raw: Any = event["data"].get("output")
        artifact = getattr(raw, "artifact", None)
        if isinstance(artifact, ToolOutput):
            output: ToolOutput | str | None = artifact
        elif isinstance(raw, (ToolOutput, str)) or raw is None:
            output = raw
        else:
            output = str(raw)
        return ToolEndEvent(
            run_id=event["run_id"],
            tool_name=event["name"],
            output=output,
        )
    if kind == "on_custom_event" and event["name"] == _TOOL_PROGRESS:
        return ToolProgressEvent(
            parent_ids=list(event.get("parent_ids", [])),
            step=event["data"].get("step", ""),
        )
    return None
