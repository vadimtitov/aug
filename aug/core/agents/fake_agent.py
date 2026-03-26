"""Fake agent — hardcoded response, no LLM, no tools.

Useful for testing the full request/response wiring without a live LLM.
Overrides build() to use a simple single-node graph — no agentic loop needed.
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from aug.core.agents.base_agent import BaseAgent
from aug.core.state import AgentState, AgentStateUpdate

_FAKE_RESPONSE = "[AUG] Hello from the fake agent. No LLM connected."


class FakeAgent(BaseAgent):
    async def respond(self, state: AgentState) -> AgentStateUpdate:
        return AgentStateUpdate(messages=[AIMessage(content=_FAKE_RESPONSE)])

    def build(self, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("respond", self.respond)
        graph.set_entry_point("respond")
        graph.add_edge("respond", END)
        return graph.compile(checkpointer=checkpointer)
