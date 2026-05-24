"""Subagent tool factory.

Wraps a BaseAgent instance as a @tool so the main agent can delegate
focused tasks to an isolated agentic loop.  The subagent runs without a
checkpointer (no history persistence) and with a lower recursion limit.

Usage (in registry.py):
    run_subagent = make_run_subagent_tool(_subagent_instance)
    _V11_TOOLS = [*_V10_TOOLS, run_subagent]
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError

from aug.core.agents.base_agent import BaseAgent
from aug.core.events import (
    ChatModelStreamEvent,
    ToolProgressEvent,
    ToolStartEvent,
    send_subagent_tool_start,
    send_tool_progress_update,
)

logger = logging.getLogger(__name__)


def make_run_subagent_tool(subagent: BaseAgent) -> BaseTool:
    """Return a run_subagent tool backed by *subagent*.

    The tool iterates the subagent's event stream, forwards progress updates
    to the parent agent's Telegram message via send_tool_progress_update, and
    returns the accumulated final response text.
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

        final_text = ""
        try:
            async for event in subagent.arun(
                prompt,
                interface=interface,
                sender_id=sender_id,
                thread_id=thread_id,
            ):
                match event:
                    case ChatModelStreamEvent(delta=delta) if delta:
                        final_text += delta
                    case ToolStartEvent(tool_name=tool_name, args=args):
                        await send_subagent_tool_start(tool_name, args)
                    case ToolProgressEvent(step=s) if s:
                        await send_tool_progress_update(s)
        except GraphRecursionError:
            suffix = "\n\n⚠️ Subagent hit the step limit without finishing."
            return (final_text + suffix) if final_text else suffix.strip()
        except Exception as exc:
            logger.error("run_subagent failed: %s", exc)
            return f"Subagent failed: {exc}"

        return final_text or "Subagent produced no text response."

    return run_subagent
