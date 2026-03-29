"""General-purpose configurable chat agent."""

import logging
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from aug.core.agents.base_agent import BaseAgent
from aug.core.llm import build_chat_model
from aug.core.prompts import INTERFACE_PROMPTS, build_system_prompt
from aug.core.reflexes import Reflex
from aug.core.state import AgentState, AgentStateUpdate
from aug.utils.logging import log_token_usage

logger = logging.getLogger(__name__)


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
        reflexes: list[Reflex] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.tools = tools or []
        self.reflexes = reflexes or []
        self._system_prompt = system_prompt
        self._model_name = model
        self._llm = build_chat_model(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            seed=seed,
        ).bind_tools(self.tools)

    def _build_system_prompt(self, state: AgentState) -> str:
        prompts = INTERFACE_PROMPTS.get(state.interface)
        interface_context = prompts.interface_context if prompts else ""
        parts = [self._system_prompt, interface_context]
        return "\n\n".join(p for p in parts if p)

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        return AgentStateUpdate(system_prompt=self._build_system_prompt(state))

    async def respond(self, state: AgentState) -> AgentStateUpdate:
        messages = _drop_orphaned_tool_calls(state.messages)
        if state.system_prompt:
            messages = [SystemMessage(content=state.system_prompt), *messages]
        logger.debug("llm_call model=%s messages=%d", self._model_name, len(messages))
        response: AIMessage = await self._llm.ainvoke(messages)
        log_token_usage(response)
        return AgentStateUpdate(messages=[response])


class TimeAwareChatAgent(ChatAgent):
    """ChatAgent that stamps incoming human messages with the current UTC time
    and injects the current time into the system prompt."""

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        prompt = self._build_system_prompt(state)
        last = state.messages[-1] if state.messages else None
        if isinstance(last, HumanMessage):
            stamped = HumanMessage(content=_stamp(last.content, now), id=last.id)
            return AgentStateUpdate(system_prompt=prompt, messages=[stamped])
        return AgentStateUpdate(system_prompt=prompt)


class AugAgent(BaseAgent):
    """Personal assistant agent using the full AUG system prompt.

    Reads identity, user knowledge, and memory from data/memory/ and builds
    the system prompt via build_system_prompt(). Time-aware: stamps incoming
    human messages with the current UTC time.
    """

    def __init__(
        self,
        model: str,
        *,
        tools: list[BaseTool] | None = None,
        reflexes: list[Reflex] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        seed: int | None = None,
        recursion_limit: int = 25,
    ) -> None:
        super().__init__()
        self.tools = tools or []
        self.reflexes = reflexes or []
        self.recursion_limit = recursion_limit
        self._model_name = model
        self._llm = build_chat_model(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            seed=seed,
        ).bind_tools(self.tools)

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        prompt = build_system_prompt(state)
        last = state.messages[-1] if state.messages else None
        if isinstance(last, HumanMessage):
            stamped = HumanMessage(content=_stamp(last.content, now), id=last.id)
            return AgentStateUpdate(system_prompt=prompt, messages=[stamped])
        return AgentStateUpdate(system_prompt=prompt)

    async def respond(self, state: AgentState) -> AgentStateUpdate:
        messages = _drop_orphaned_tool_calls(state.messages)
        if state.system_prompt:
            messages = [SystemMessage(content=state.system_prompt), *messages]
        logger.debug("llm_call model=%s messages=%d", self._model_name, len(messages))
        response: AIMessage = await self._llm.ainvoke(messages)
        log_token_usage(response)
        return AgentStateUpdate(messages=[response])


def _stamp(content: str | list, now: str) -> str | list:
    """Prepend a timestamp to a message without destroying multimodal content."""
    if isinstance(content, str):
        return f"[{now}] {content}"
    return [{"type": "text", "text": f"[{now}]"}, *content]


def _drop_orphaned_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove AIMessages whose tool calls were never answered.

    If a tool raises an unhandled exception the graph aborts before writing
    the ToolMessages, leaving the state with a dangling tool_calls entry that
    causes every subsequent LLM call to fail with a 400.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    result = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if not all(tc["id"] in answered for tc in msg.tool_calls):
                continue
        result.append(msg)
    return result
