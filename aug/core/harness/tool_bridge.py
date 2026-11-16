"""Running an AUG tool for an agent CLI (Claude Code), whatever shape the CLI wants.

The LangChain tool runs with the turn's ToolContext in its config, exactly as on
LangGraph, so the tools themselves don't know which harness runs them.  Its result
comes back as text and images (``ToolOutcome``); each harness puts those in its own
CLI's shape.
"""

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from aug.core.harness.tool_context import ToolContext
from aug.core.prompts import HARNESS_TOOL_ERROR_RESULT, TOOL_NO_OUTPUT
from aug.utils.logging import set_context_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolImage:
    media_type: str
    data: str  # base64


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool call gave the model: text and images, and whether it failed."""

    blocks: tuple[str | ToolImage, ...]
    error: bool = False


def input_schema(tool: BaseTool) -> dict[str, Any]:
    """*tool*'s arguments as JSON schema."""
    schema = tool.tool_call_schema
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


async def invoke_tool(
    tool: BaseTool, args: Any, call_id: str | None, ctx: ToolContext | None
) -> ToolOutcome:
    """Run *tool* for the CLI's call *call_id*, its events filed under that call.
    Never raises but for cancellation: a tool that raised is a failed outcome."""
    config: RunnableConfig = {}
    if ctx is not None:
        set_context_id(ctx.context_id)
        config = (ctx.under(call_id) if call_id else ctx).runnable_config()
    call = {"type": "tool_call", "id": call_id or uuid.uuid4().hex, "name": tool.name}
    try:
        result = await tool.ainvoke(
            {**call, "args": args if isinstance(args, dict) else {}}, config
        )
    except Exception as e:
        logger.warning("tool %s raised: %s: %s", tool.name, type(e).__name__, e)
        return ToolOutcome(blocks=(HARNESS_TOOL_ERROR_RESULT.format(error=e),), error=True)
    if isinstance(result, ToolMessage):
        return ToolOutcome(blocks=result_blocks(result.content), error=result.status == "error")
    return ToolOutcome(blocks=result_blocks(result))


def result_blocks(content: Any) -> tuple[str | ToolImage, ...]:
    """A LangChain tool result as text and images; never empty."""
    if isinstance(content, str):
        return (content or TOOL_NO_OUTPUT,)
    if not isinstance(content, list):
        return (str(content),)
    blocks: list[str | ToolImage] = []
    for block in content:
        if isinstance(block, str):
            blocks.append(block)
        elif not isinstance(block, dict):
            blocks.append(str(block))
        elif block.get("type") == "text":
            blocks.append(str(block.get("text", "")))
        elif (image := _image(block)) is not None:
            blocks.append(image)
        else:
            blocks.append(json.dumps(block, default=str))
    return tuple(blocks) or (TOOL_NO_OUTPUT,)


def _image(block: dict) -> ToolImage | None:
    """A LangChain image block as a ToolImage, or None if it isn't one."""
    kind = block.get("type")
    if kind == "image":
        data = block.get("base64") or block.get("data")
        mime = block.get("mime_type") or block.get("mimeType")
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode()
        if data and mime:
            return ToolImage(media_type=str(mime), data=str(data))
    if kind == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            mime, data = url[5:].split(";base64,", 1)
            return ToolImage(media_type=mime, data=data)
    return None
