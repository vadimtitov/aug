"""AUG's tools on Claude Code, and the guard on Claude's own file and shell tools.

AUG's tools (LangChain tools, the MCP manager's servers included) reach Claude through
one in-process MCP server, ``aug``.  Each call runs the LangChain tool with the turn's
ToolContext in its config, exactly as on LangGraph, so the tools themselves don't know
which harness runs them.  Claude sees them as ``mcp__aug__<name>``; events carry the
canonical ``<name>``.

Claude's own Bash, Read, Write, Edit, Glob and Grep pass a PreToolUse hook first: it
refuses the paths no tool may touch (``blocked_paths``) and rewrites every shell command
to run through hushed, as ``run_bash`` does.  Monitor's commands are checked too.
"""

import json
import logging
import uuid
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    HookContext,
    HookMatcher,
    McpSdkServerConfig,
    SdkMcpTool,
    create_sdk_mcp_server,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from aug.core.harness.tool_context import TOOL_CONTEXT_CONFIG_KEY, ToolContext
from aug.core.harness.types import AgentEvent, StatusUpdate
from aug.core.prompts import CLAUDE_TOOL_ERROR_RESULT, TOOL_NO_OUTPUT
from aug.core.tools.blocked_paths import command_block_reason, path_block_reason
from aug.core.tools.run_bash import hushed_command
from aug.utils.logging import set_context_id

logger = logging.getLogger(__name__)

AUG_SERVER = "aug"
_PREFIX = f"mcp__{AUG_SERVER}__"
# Claude's tools that read or write files, and the input field holding the path.
_PATH_FIELDS = ("file_path", "path", "notebook_path")
GUARDED_TOOLS = ("Bash", "Monitor", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep")


def canonical_name(name: str) -> str:
    """The name AUG uses for a tool Claude calls: ``mcp__aug__run_ssh`` is ``run_ssh``."""
    return name.removeprefix(_PREFIX)


def claude_name(name: str) -> str:
    return f"{_PREFIX}{name}"


class ToolCalls:
    """Tool calls of one chat: which ToolContext AUG's tools get, and which tool use
    each MCP call belongs to.

    The MCP handler is not told its tool_use_id, so it is matched to the ToolUseBlock
    that announced it, by name and arguments.  Unmatched (a race, or two identical
    calls answered out of order) only loses the parent of the tool's own events.
    """

    def __init__(self) -> None:
        self.context: ToolContext | None = None
        self._announced: dict[tuple[str, str], deque[str]] = defaultdict(deque)

    def announce(self, tool_use_id: str, name: str, args: dict) -> None:
        if name.startswith(_PREFIX):
            self._announced[(canonical_name(name), _args_key(args))].append(tool_use_id)

    def claim(self, name: str, args: dict) -> str | None:
        queue = self._announced.get((name, _args_key(args)))
        return queue.popleft() if queue else None

    def clear(self) -> None:
        self._announced.clear()


def aug_mcp_server(tools: Sequence[BaseTool], calls: ToolCalls) -> McpSdkServerConfig:
    """One in-process MCP server with every tool in *tools*."""
    return create_sdk_mcp_server(AUG_SERVER, tools=[_adapt(t, calls) for t in tools])


def tool_fingerprint(tools: Sequence[BaseTool]) -> list[dict[str, Any]]:
    """What Claude is told about *tools*: a change means a new process."""
    return [
        {"name": t.name, "description": t.description, "schema": _input_schema(t)}
        for t in sorted(tools, key=lambda t: t.name)
    ]


def guard_hooks() -> dict:
    """The PreToolUse hook on Claude's own file and shell tools."""
    return {"PreToolUse": [HookMatcher(matcher="|".join(GUARDED_TOOLS), hooks=[guard_tool_use])]}


async def guard_tool_use(hook_input: Any, tool_use_id: str | None, context: HookContext) -> dict:
    """Refuse blocked paths; run shell commands through hushed."""
    name = hook_input.get("tool_name", "")
    args = hook_input.get("tool_input") or {}
    if name == "Bash":
        command = args.get("command", "")
        if reason := command_block_reason(command):
            return _deny(reason)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {**args, "command": hushed_command(command)},
            }
        }
    if name == "Monitor":
        # Its output streams line by line, which hushed would hold back until the
        # command exits: checked, not rewritten, so it runs without secrets.
        command = args.get("command")
        if isinstance(command, str) and (reason := command_block_reason(command)):
            return _deny(reason)
        return {}
    for field in _PATH_FIELDS:
        value = args.get(field)
        if isinstance(value, str) and value and (reason := path_block_reason(Path(value))):
            return _deny(reason)
    return {}


def mcp_content(content: Any) -> list[dict[str, Any]]:
    """A LangChain tool result as MCP content blocks: text and images."""
    if isinstance(content, str):
        return [{"type": "text", "text": content or TOOL_NO_OUTPUT}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
        elif not isinstance(block, dict):
            blocks.append({"type": "text", "text": str(block)})
        elif block.get("type") == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif (image := _image(block)) is not None:
            blocks.append(image)
        else:
            blocks.append({"type": "text", "text": json.dumps(block, default=str)})
    return blocks or [{"type": "text", "text": TOOL_NO_OUTPUT}]


def _image(block: dict) -> dict | None:
    """An MCP image block from a LangChain image block, or None if it isn't one."""
    kind = block.get("type")
    if kind == "image":
        data = block.get("base64") or block.get("data")
        mime = block.get("mime_type") or block.get("mimeType")
        if data and mime:
            return {"type": "image", "data": data, "mimeType": mime}
    if kind == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            mime, data = url[5:].split(";base64,", 1)
            return {"type": "image", "data": data, "mimeType": mime}
    return None


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _adapt(tool: BaseTool, calls: ToolCalls) -> SdkMcpTool[Any]:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        tool_use_id = calls.claim(tool.name, args)
        ctx = calls.context
        config: dict[str, Any] = {}
        if ctx is not None:
            set_context_id(ctx.context_id)
            if tool_use_id is not None:
                ctx = _under(ctx, tool_use_id)
            config = {
                "configurable": {
                    "thread_id": ctx.context_id,
                    "interface": ctx.turn.reply_to.interface,
                    "sender_id": ctx.turn.reply_to.chat_id,
                    TOOL_CONTEXT_CONFIG_KEY: ctx,
                }
            }
        call = {"type": "tool_call", "id": tool_use_id or uuid.uuid4().hex, "name": tool.name}
        try:
            result = await tool.ainvoke({**call, "args": args}, config)
        except Exception as e:
            logger.warning("tool %s raised: %s: %s", tool.name, type(e).__name__, e)
            error = CLAUDE_TOOL_ERROR_RESULT.format(error=e)
            return {"content": [{"type": "text", "text": error}], "is_error": True}
        if isinstance(result, ToolMessage):
            return {"content": mcp_content(result.content), "is_error": result.status == "error"}
        return {"content": mcp_content(result)}

    return SdkMcpTool(
        name=tool.name,
        description=tool.description,
        input_schema=_input_schema(tool),
        handler=handler,
    )


def _under(ctx: ToolContext, tool_use_id: str) -> ToolContext:
    """*ctx* with every event it emits filed under the tool call *tool_use_id*."""
    emit = ctx._emit

    async def emit_under(event: AgentEvent) -> None:
        if emit is None:
            return
        if not isinstance(event, StatusUpdate) and event.parent_id is None:
            event = replace(event, parent_id=tool_use_id)
        await emit(event)

    return replace(ctx, _emit=emit_under)


def _input_schema(tool: BaseTool) -> dict[str, Any]:
    schema = tool.tool_call_schema
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


def _args_key(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)
