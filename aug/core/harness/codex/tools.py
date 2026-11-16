"""AUG's tools on Codex: dynamic tools, answered in AUG's own process.

A Codex thread is started with AUG's tools (LangChain tools, the MCP manager's included)
as dynamic tools: the profile's ``always_loaded`` ones plainly, the rest in the ``aug``
namespace, deferred until Codex looks them up.  Codex asks AUG to run each call
(``item/tool/call``); ``invoke_tool`` runs it as on Claude Code, and its outcome goes
back as Codex content items.  Events carry the tool's own name.

A thread keeps the tools it started with: Codex stores them with it.  Tools added
later (an MCP server installed) reach a chat's next session; a call to a tool that has
since gone answers that it is gone.
"""

import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from aug.core.harness.tool_bridge import ToolOutcome, input_schema, invoke_tool
from aug.core.harness.tool_context import ToolContext
from aug.core.prompts import CODEX_AUG_NAMESPACE_DESCRIPTION, CODEX_TOOL_GONE_RESULT
from aug.utils.codex_app_server import JsonObject

logger = logging.getLogger(__name__)

AUG_NAMESPACE = "aug"  # the namespace AUG's deferred tools are in


def dynamic_tools(tools: Sequence[BaseTool], always_loaded: Collection[str]) -> list[JsonObject]:
    """*tools* as thread/start's dynamicTools: those in *always_loaded* plainly, the
    rest deferred in the AUG_NAMESPACE namespace."""
    specs = [_spec(t) for t in tools if t.name in always_loaded]
    deferred = [{**_spec(t), "deferLoading": True} for t in tools if t.name not in always_loaded]
    if deferred:
        specs.append(
            {
                "type": "namespace",
                "name": AUG_NAMESPACE,
                "description": CODEX_AUG_NAMESPACE_DESCRIPTION,
                "tools": deferred,
            }
        )
    return specs


async def run_tool(
    tools: Mapping[str, BaseTool], name: str, args: Any, call_id: str, ctx: ToolContext | None
) -> JsonObject:
    """Run the AUG tool *name* for Codex's call *call_id*; returns the call's answer.
    Never raises but for cancellation."""
    tool = tools.get(name)
    if tool is None:
        logger.warning("codex called a tool AUG no longer has: %s", name)
        return tool_answer(
            ToolOutcome(blocks=(CODEX_TOOL_GONE_RESULT.format(name=name),), error=True)
        )
    return tool_answer(await invoke_tool(tool, args, call_id, ctx))


def tool_answer(outcome: ToolOutcome) -> JsonObject:
    """*outcome* as the answer to Codex's ``item/tool/call``: text and images."""
    items = [
        {"type": "inputText", "text": block}
        if isinstance(block, str)
        else {"type": "inputImage", "imageUrl": f"data:{block.media_type};base64,{block.data}"}
        for block in outcome.blocks
    ]
    return {"contentItems": items, "success": not outcome.error}


def items_text(items: Any) -> str | None:
    """What a finished dynamic tool call said, as text for ToolEnded."""
    if not isinstance(items, list):
        return None
    parts = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "inputText":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, dict) and item.get("type") == "inputImage":
            parts.append("[image]")
    return "\n".join(parts)


def _spec(tool: BaseTool) -> JsonObject:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "inputSchema": input_schema(tool),
    }
