"""General-purpose configurable chat agent."""

import asyncio
import base64
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from aug.core.agents.base_agent import BaseAgent
from aug.core.compaction import compact_messages, count_tokens
from aug.core.llm import build_chat_model
from aug.core.prompts import IMAGE_DESCRIPTION_SYSTEM_NOTE, INTERFACE_PROMPTS, build_system_prompt
from aug.core.reflexes import Reflex
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.describe_image import make_describe_image_tool
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
        vision_description_model: str | None = None,
        compaction_model: str | None = None,
        compaction_threshold: float = 0.80,
        context_window: int = 200_000,
        max_summary_tokens: int = 2000,
    ) -> None:
        super().__init__()
        self._vision_description_model = vision_description_model
        self.tools = list(tools or [])
        if vision_description_model:
            self.tools.append(make_describe_image_tool(vision_description_model))
        self.reflexes = reflexes or []
        self._system_prompt = system_prompt
        self._model_name = model
        self._compaction_model = compaction_model
        self._compaction_threshold = compaction_threshold
        self._context_window = context_window
        self._max_summary_tokens = max_summary_tokens
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
        state_changes: list[AnyMessage] = []
        tokens = count_tokens(messages)
        threshold = int(self._compaction_threshold * self._context_window)
        logger.debug("context tokens=%d threshold=%d", tokens, threshold)
        if self._compaction_model and tokens > threshold:
            messages, state_changes = await compact_messages(
                messages,
                self._compaction_model,
                context_window=self._context_window,
                max_summary_tokens=self._max_summary_tokens,
            )
        if state.system_prompt:
            messages = [SystemMessage(content=state.system_prompt), *messages]
        if not self._vision_description_model:
            messages = await _expand_images(messages)
        logger.debug("llm_call model=%s messages=%d", self._model_name, len(messages))
        response: AIMessage = await self._llm.ainvoke(messages)
        log_token_usage(response)
        return AgentStateUpdate(messages=[*state_changes, response])


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
        vision_description_model: str | None = None,
        compaction_model: str | None = None,
        compaction_threshold: float = 0.80,
        context_window: int = 200_000,
        max_summary_tokens: int = 2000,
    ) -> None:
        super().__init__()
        self._vision_description_model = vision_description_model
        self.tools = list(tools or [])
        if vision_description_model:
            self.tools.append(make_describe_image_tool(vision_description_model))
        self.reflexes = reflexes or []
        self.recursion_limit = recursion_limit
        self._model_name = model
        self._compaction_model = compaction_model
        self._compaction_threshold = compaction_threshold
        self._context_window = context_window
        self._max_summary_tokens = max_summary_tokens
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
        if self._vision_description_model:
            prompt = f"{prompt}\n\n{IMAGE_DESCRIPTION_SYSTEM_NOTE}"
        last = state.messages[-1] if state.messages else None
        if isinstance(last, HumanMessage):
            stamped = HumanMessage(content=_stamp(last.content, now), id=last.id)
            return AgentStateUpdate(system_prompt=prompt, messages=[stamped])
        return AgentStateUpdate(system_prompt=prompt)

    async def respond(self, state: AgentState) -> AgentStateUpdate:
        messages = _drop_orphaned_tool_calls(state.messages)
        state_changes: list[AnyMessage] = []
        tokens = count_tokens(messages)
        threshold = int(self._compaction_threshold * self._context_window)
        logger.debug("context tokens=%d threshold=%d", tokens, threshold)
        if self._compaction_model and tokens > threshold:
            messages, state_changes = await compact_messages(
                messages,
                self._compaction_model,
                context_window=self._context_window,
                max_summary_tokens=self._max_summary_tokens,
            )
        if state.system_prompt:
            messages = [SystemMessage(content=state.system_prompt), *messages]
        if not self._vision_description_model:
            messages = await _expand_images(messages)
        logger.debug("llm_call model=%s messages=%d", self._model_name, len(messages))
        response: AIMessage = await self._llm.ainvoke(messages)
        log_token_usage(response)
        return AgentStateUpdate(messages=[*state_changes, response])


def _stamp(content: str | list, now: str) -> str | list:
    """Prepend a timestamp to a message without destroying multimodal content."""
    if isinstance(content, str):
        return f"[{now}] {content}"
    return [{"type": "text", "text": f"[{now}]"}, *content]


_IMG_TAG = re.compile(r"\[\[img:([^|]+)\|([^\]]+)\]\]")


async def _expand_images(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Re-inline [[img:path|mime]] markers in the last HumanMessage as image_url blocks.

    Markers in historical messages are left as plain text so the LLM understands
    an image was present without re-sending the binary data on every turn.
    """
    last_human_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=None,
    )
    if last_human_idx is None:
        return messages
    msg = messages[last_human_idx]
    blocks = await _inline_image_markers(msg.content)
    if blocks is None:
        return messages
    result = list(messages)
    result[last_human_idx] = HumanMessage(content=blocks, id=msg.id)
    return result


async def _inline_image_markers(content: str | list) -> list | None:
    """Expand [[img:path|mime]] markers into image_url blocks.

    Returns None when no markers are found (caller can skip the copy).
    """
    if isinstance(content, str):
        if not _IMG_TAG.search(content):
            return None
        return await _markers_to_blocks(content)
    if isinstance(content, list):
        expanded = []
        changed = False
        for block in content:
            if block.get("type") == "text" and _IMG_TAG.search(block["text"]):
                expanded.extend(await _markers_to_blocks(block["text"]))
                changed = True
            else:
                expanded.append(block)
        return expanded if changed else None
    return None


async def _markers_to_blocks(text: str) -> list:
    """Split a string containing [[img:path|mime]] markers into content blocks."""
    blocks = []
    pos = 0
    for m in _IMG_TAG.finditer(text):
        before = text[pos : m.start()].strip()
        if before:
            blocks.append({"type": "text", "text": before})
        path, mime_type = m.group(1), m.group(2)
        try:
            data = await asyncio.to_thread(Path(path).read_bytes)
            encoded = base64.b64encode(data).decode()
            blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
            )
            blocks.append({"type": "text", "text": f"[Image: {path}]"})
        except FileNotFoundError:
            blocks.append({"type": "text", "text": f"[Image not found: {path}]"})
        pos = m.end()
    after = text[pos:].strip()
    if after:
        blocks.append({"type": "text", "text": after})
    return blocks


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
