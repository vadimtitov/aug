"""AUG's tools on Claude Code.

AUG's tools (LangChain tools, the MCP manager's servers included) reach Claude through
two in-process MCP servers: ``aug_core`` holds the ones the profile keeps loaded in
every request, ``aug`` the rest, which wait behind Claude's ToolSearch.  Each call runs
the LangChain tool through ``tool_bridge``.  Claude sees them as ``mcp__aug_core__<name>``
or ``mcp__aug__<name>``; events carry the canonical ``<name>``.
"""

import json
from collections import defaultdict, deque
from collections.abc import Collection, Sequence
from typing import Any

from claude_agent_sdk import McpServerConfig, SdkMcpTool, create_sdk_mcp_server
from langchain_core.tools import BaseTool

from aug.core.harness.tool_bridge import ToolImage, ToolOutcome, input_schema, invoke_tool
from aug.core.harness.tool_context import ToolContext

AUG_SERVER = "aug"  # AUG's tools Claude loads through ToolSearch when it needs them
AUG_CORE_SERVER = "aug_core"  # those the profile keeps loaded in every request
AUG_SERVERS = (AUG_CORE_SERVER, AUG_SERVER)
_PREFIXES = tuple(f"mcp__{server}__" for server in AUG_SERVERS)


def canonical_name(name: str) -> str:
    """The name AUG uses for a tool Claude calls: ``mcp__aug__run_ssh`` is ``run_ssh``,
    ``mcp__aug_core__note`` is ``note``."""
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def is_aug_tool(name: str) -> bool:
    return name.startswith(_PREFIXES)


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
        if is_aug_tool(name):
            self._announced[(canonical_name(name), _args_key(args))].append(tool_use_id)

    def claim(self, name: str, args: dict) -> str | None:
        queue = self._announced.get((name, _args_key(args)))
        return queue.popleft() if queue else None

    def clear(self) -> None:
        self._announced.clear()


def aug_mcp_servers(
    tools: Sequence[BaseTool], always_loaded: Collection[str], calls: ToolCalls
) -> dict[str, McpServerConfig]:
    """AUG's two in-process MCP servers: the tools named in *always_loaded* on one Claude
    always loads, every other tool on one it defers behind ToolSearch."""
    core = create_sdk_mcp_server(
        AUG_CORE_SERVER, tools=[_adapt(t, calls) for t in tools if t.name in always_loaded]
    )
    rest = create_sdk_mcp_server(
        AUG_SERVER, tools=[_adapt(t, calls) for t in tools if t.name not in always_loaded]
    )
    # The SDK hands a server's config keys, all but the instance, to the CLI.
    return {AUG_CORE_SERVER: {**core, "alwaysLoad": True}, AUG_SERVER: rest}  # type: ignore[dict-item]


def tool_fingerprint(tools: Sequence[BaseTool]) -> list[dict[str, Any]]:
    """What Claude is told about *tools*: a change means a new process."""
    return [
        {"name": t.name, "description": t.description, "schema": input_schema(t)}
        for t in sorted(tools, key=lambda t: t.name)
    ]


def mcp_content(outcome: ToolOutcome) -> dict[str, Any]:
    """A tool call's outcome as an MCP tool result."""
    content = [
        {"type": "image", "data": b.data, "mimeType": b.media_type}
        if isinstance(b, ToolImage)
        else {"type": "text", "text": b}
        for b in outcome.blocks
    ]
    return {"content": content, "is_error": outcome.error}


def _adapt(tool: BaseTool, calls: ToolCalls) -> SdkMcpTool[Any]:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        tool_use_id = calls.claim(tool.name, args)
        return mcp_content(await invoke_tool(tool, args, tool_use_id, calls.context))

    return SdkMcpTool(
        name=tool.name,
        description=tool.description,
        input_schema=input_schema(tool),
        handler=handler,
    )


def _args_key(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)
