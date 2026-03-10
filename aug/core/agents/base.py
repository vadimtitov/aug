"""Base class for all AUG agents."""

from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from aug.core.state import AgentState, AgentStateUpdate


class BaseAgent(ABC):
    """All agents inherit from this class.

    The default ``build()`` provides a full agentic loop:

        [call_model] --has tool calls--> [call_tools] --back--> [call_model]
              |
           no tool calls
              |
             END

    To add a new agent:
    1. Subclass BaseAgent in ``aug/core/agents/<name>.py``.
    2. Define ``tools`` — list of @tool functions available to the LLM.
    3. Implement ``respond()`` — calls the LLM and returns its response.
    4. Register an instance in ``aug/core/graph.py``.
    """

    tools: list[BaseTool] = []

    @abstractmethod
    def respond(self, state: AgentState) -> AgentStateUpdate:
        """Call the LLM and return its response (including any tool call requests)."""

    def _call_tools(self, state: AgentState) -> AgentStateUpdate:
        """Execute any tool calls requested by the LLM."""
        tools_by_name = {t.name: t for t in self.tools}
        last_msg: AIMessage = state["messages"][-1]  # type: ignore[assignment]
        results: list[ToolMessage] = []
        for tool_call in last_msg.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            output = tool.invoke(tool_call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        return {"messages": results}

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "call_tools"
        return END

    def build(self, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("call_model", self.respond)
        graph.add_node("call_tools", self._call_tools)
        graph.set_entry_point("call_model")
        graph.add_conditional_edges("call_model", self._should_continue)
        graph.add_edge("call_tools", "call_model")
        return graph.compile(checkpointer=checkpointer)
