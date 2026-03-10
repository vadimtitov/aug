"""Default agent — real LLM via LiteLLM proxy with tool support."""

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from aug.core.agents.base import BaseAgent
from aug.core.llm import build_llm
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.datetime_tool import get_current_datetime


class DefaultAgent(BaseAgent):
    tools: list[BaseTool] = [get_current_datetime]

    def respond(self, state: AgentState) -> AgentStateUpdate:
        llm = build_llm({"model": "gpt-4o", "temperature": 0.7}).bind_tools(self.tools)
        response: AIMessage = llm.invoke(state["messages"])
        return {"messages": [response]}
