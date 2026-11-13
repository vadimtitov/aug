"""LangGraph's stream events, translated into harness events.

LangChain's ``astream_events`` yields untyped dicts.  ``parse_event`` turns the ones AUG
uses into harness events and drops the rest; nothing outside the LangGraph harness and
its agents sees a raw LangGraph event.

A tool's ``ToolContext.emit()`` travels as a LangChain custom event, so LangGraph stamps
it with the run_id of the tool that emitted it.  That run_id is the tool's ToolStarted
id, which makes it the event's parent.
"""

from dataclasses import replace
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.schema import StreamEvent

from aug.core.harness.types import AgentEvent, StatusUpdate, TextDelta, ToolEnded, ToolStarted
from aug.core.tools.output import ToolOutput

_AUG_EVENT = "aug_event"


async def dispatch(event: AgentEvent) -> None:
    """Send *event* up the running graph's event stream.  A no-op outside a run."""
    try:
        await adispatch_custom_event(_AUG_EVENT, {"event": event})
    except RuntimeError:
        pass  # no active run context (e.g. a tool called directly in a unit test)


def parse_event(event: StreamEvent) -> AgentEvent | None:
    kind = event["event"]
    if kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        return TextDelta(text=chunk.content if isinstance(chunk.content, str) else "")
    if kind == "on_tool_start":
        return ToolStarted(
            id=event["run_id"],
            name=event["name"],
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
        return ToolEnded(
            id=event["run_id"],
            name=event["name"],
            output=output,
            error=getattr(raw, "status", None) == "error",
        )
    if kind == "on_custom_event" and event["name"] == _AUG_EVENT:
        emitted = event["data"]["event"]
        # Status comes from outside any tool (compaction runs in a graph node).
        if (
            not isinstance(emitted, StatusUpdate)
            and emitted.parent_id is None
            and event.get("run_id")
        ):
            emitted = replace(emitted, parent_id=event["run_id"])
        return emitted
    return None
