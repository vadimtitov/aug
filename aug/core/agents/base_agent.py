"""Base class for all AUG agents."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from aug.core.events import AgentEvent, ToolEndEvent, ToolStartEvent, parse_event
from aug.core.state import AgentState, AgentStateUpdate

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """All agents inherit from this class.

    The default ``build()`` provides a full agentic loop:

        [preprocess] -> [call_model] --has tool calls--> [call_tools] --back--> [call_model]
                               |
                            no tool calls
                               |
                        [postprocess] -> END

    Hooks:
        preprocess()  — override to modify state before the LLM is called
                        (inject system prompt, add tools, annotate messages, etc.)
                        Default: no-op.
        respond()     — abstract, must be implemented. Calls the LLM.
        postprocess() — override to transform/log the final response.
                        Default: no-op.

    To add a new agent:
    1. Subclass BaseAgent in ``aug/core/agents/<name>.py``.
    2. Define ``tools`` — list of @tool functions available to the LLM.
    3. Implement ``respond()`` — calls the LLM and returns its response.
    4. Optionally override ``preprocess()`` and/or ``postprocess()``.
    5. Register an instance in ``aug/core/registry.py``.
    """

    tools: list[BaseTool] = []
    recursion_limit: int = 25

    def __init__(self) -> None:
        self._compiled_graph = None

    async def astream_events(
        self,
        state: AgentState,
        config: RunnableConfig,
        checkpointer: BaseCheckpointSaver,
    ) -> AsyncIterator[AgentEvent]:
        if self._compiled_graph is None:
            self._compiled_graph = self._build(checkpointer)
        full_config: RunnableConfig = {"recursion_limit": self.recursion_limit, **config}
        raw_stream = self._compiled_graph.astream_events(state, config=full_config, version="v2")
        tool_start_times: dict[str, float] = {}
        async for raw in raw_stream:
            event = parse_event(raw)
            if event is None:
                continue
            match event:
                case ToolStartEvent(run_id=run_id, tool_name=tool_name, args=args):
                    tool_start_times[run_id] = time.monotonic()
                    logger.info("tool_start tool=%s args=%.120r", tool_name, args)
                case ToolEndEvent(run_id=run_id, tool_name=tool_name, error=error):
                    elapsed = time.monotonic() - tool_start_times.pop(run_id, time.monotonic())
                    if error:
                        logger.warning(
                            "tool_end tool=%s duration=%.2fs error=True", tool_name, elapsed
                        )
                    else:
                        logger.info("tool_end tool=%s duration=%.2fs", tool_name, elapsed)
            yield event

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        """Prepare state before the LLM is called."""
        return AgentStateUpdate()

    @abstractmethod
    def respond(self, state: AgentState) -> AgentStateUpdate:
        """Call the LLM and return its response (including any tool call requests)."""

    def postprocess(self, state: AgentState) -> AgentStateUpdate:
        """Transform or log the final LLM response before returning to the user."""
        return AgentStateUpdate()

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state.messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "call_tools"
        return "postprocess"

    def _build(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(AgentState)
        graph.add_node("preprocess", self.preprocess)
        graph.add_node("call_model", self.respond)
        graph.add_node("call_tools", ToolNode(self.tools))
        graph.add_node("postprocess", self.postprocess)
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "call_model")
        graph.add_conditional_edges("call_model", self._should_continue)
        graph.add_edge("call_tools", "call_model")
        graph.add_edge("postprocess", END)
        return graph.compile(checkpointer=checkpointer)
