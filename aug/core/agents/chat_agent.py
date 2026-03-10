"""General-purpose configurable chat agent."""

from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from aug.core.agents.base_agent import BaseAgent
from aug.core.llm import build_chat_model
from aug.core.state import AgentState, AgentStateUpdate


class ChatAgent(BaseAgent):
    """A configurable chat agent backed by any LiteLLM-compatible model.

    All parameters are set at instantiation, making it easy to define
    multiple agent variants in the registry without subclassing.

    Example::

        ChatAgent(
            model="gpt-4o",
            system_prompt="You are a concise assistant.",
            tools=[search, calculator],
            temperature=0.2,
        )
    """

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str = "",
        tools: list[BaseTool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.tools = tools or []
        self._system_prompt = system_prompt
        self._llm = build_chat_model(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            seed=seed,
        ).bind_tools(self.tools)

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        return AgentStateUpdate(system_prompt=self._system_prompt)

    def respond(self, state: AgentState) -> AgentStateUpdate:
        messages = state.messages
        if state.system_prompt:
            messages = [SystemMessage(content=state.system_prompt), *messages]
        response: AIMessage = self._llm.invoke(messages)
        return AgentStateUpdate(messages=[response])


class TimeAwareChatAgent(ChatAgent):
    """ChatAgent that stamps incoming human messages with the current UTC time
    and injects the current time into the system prompt."""

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        last = state.messages[-1] if state.messages else None
        if isinstance(last, HumanMessage):
            stamped = HumanMessage(content=f"[{now}] {last.content}", id=last.id)
            return AgentStateUpdate(system_prompt=self._system_prompt, messages=[stamped])
        return AgentStateUpdate(system_prompt=self._system_prompt)
