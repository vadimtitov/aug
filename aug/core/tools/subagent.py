"""Subagent tool factory.

Wraps a BaseAgent instance as a @tool so the main agent can delegate
focused tasks to an isolated agentic loop.  The subagent runs without a
checkpointer (no history persistence) and with a lower recursion limit.

Usage (in registry.py):
    run_subagent = make_run_subagent_tool(_subagent_instance)
    _V11_TOOLS = [*_V10_TOOLS, run_subagent]
"""

import logging
from dataclasses import replace

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError

from aug.core.agents.base_agent import BaseAgent
from aug.core.harness.types import (
    ApprovalRequested,
    ApprovalResolved,
    FileEmitted,
    TextDelta,
    ToolEnded,
    ToolProgress,
    ToolStarted,
)
from aug.core.tools.context import current_tool_context

logger = logging.getLogger(__name__)


def make_run_subagent_tool(subagent: BaseAgent) -> BaseTool:
    """Return a run_subagent tool backed by *subagent*.

    The tool iterates the subagent's event stream, re-emits its tool calls and
    progress under its own call (so the user sees them nested), passes its approval
    requests through, and returns the accumulated final response text.
    """

    @tool
    async def run_subagent(prompt: str, config: RunnableConfig) -> str:
        """Delegate a focused task to an isolated subagent and return its response.

        The subagent has full tool access and runs its own agentic loop independently
        of the current conversation — its messages are not saved to this thread.

        Use this to:
        - Parallelize independent research tasks
        - Offload long-running operations that would bloat the current context
        - Run a deep investigation without polluting the main conversation history

        Write a fully self-contained prompt: include all relevant context, because
        the subagent has no access to the current conversation history.

        Args:
            prompt: Complete task description with all context the subagent needs.
        """
        configurable = (config or {}).get("configurable") or {}
        interface: str = configurable.get("interface", "rest_api")
        sender_id: str = configurable.get("sender_id", "")
        thread_id: str = configurable.get("thread_id", "")

        ctx = current_tool_context()
        final_text = ""
        try:
            async for event in subagent.arun(
                prompt,
                interface=interface,
                sender_id=sender_id,
                thread_id=thread_id,
                tool_context=ctx,
            ):
                match event:
                    case TextDelta(text=text) if text:
                        final_text += text
                    case ToolStarted() | ToolProgress() | ToolEnded() | FileEmitted() if (
                        ctx is not None
                    ):
                        # Re-parented onto this call: the user sees one nested block, and
                        # gets the subagent's files.
                        await ctx.emit(replace(event, parent_id=None))
                    case ApprovalRequested() | ApprovalResolved() if ctx is not None:
                        await ctx.emit(event)
        except GraphRecursionError:
            suffix = "\n\n⚠️ Subagent hit the step limit without finishing."
            return (final_text + suffix) if final_text else suffix.strip()
        except Exception as exc:
            logger.error("run_subagent failed: %s", exc)
            return f"Subagent failed: {exc}"

        return final_text or "Subagent produced no text response."

    return run_subagent
