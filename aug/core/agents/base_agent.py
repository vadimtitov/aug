"""Base class for all AUG agents."""

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel import Pregel as CompiledGraph

from aug.core.state import AgentState, AgentStateUpdate


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

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        """Prepare state before the LLM is called.

        Override to inject a dynamic system prompt, expose conditional tools,
        annotate messages with metadata (e.g. timestamps), etc.
        """
        return AgentStateUpdate()

    @abstractmethod
    def respond(self, state: AgentState) -> AgentStateUpdate:
        """Call the LLM and return its response (including any tool call requests)."""

    def postprocess(self, state: AgentState) -> AgentStateUpdate:
        """Transform or log the final LLM response before returning to the user.

        Override to add response filtering, logging, formatting, etc.
        """
        return AgentStateUpdate()

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state.messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "call_tools"
        return "postprocess"

    def build(self, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
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
        compiled = graph.compile(checkpointer=checkpointer)
        return compiled.with_config({"recursion_limit": self.recursion_limit})
