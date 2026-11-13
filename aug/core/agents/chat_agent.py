"""General-purpose configurable chat agent."""

import asyncio
import base64
import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from aug.core.agents.base_agent import BaseAgent
from aug.core.compaction import compact_messages, count_tokens
from aug.core.llm import build_chat_model
from aug.core.prompts import (
    IMAGE_DESCRIPTION_SYSTEM_NOTE,
    INTERFACE_PROMPTS,
    USER_IMAGE_HINT,
    build_system_prompt,
)
from aug.core.reflexes import Reflex
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.describe_image import make_describe_image_tool
from aug.utils.logging import log_token_usage
from aug.utils.media import IMAGE_MARKER, MODEL_IMAGE_TYPES
from aug.utils.storage import file_store

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
        self.model_name = model
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
        messages = await _expand_images(messages, vision=not self._vision_description_model)
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
        # Everything but the tools, so with_extra_tools() can rebuild this agent.
        self._settings = dict(
            model=model,
            reflexes=list(reflexes) if reflexes else None,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            seed=seed,
            recursion_limit=recursion_limit,
            vision_description_model=vision_description_model,
            compaction_model=compaction_model,
            compaction_threshold=compaction_threshold,
            context_window=context_window,
            max_summary_tokens=max_summary_tokens,
        )
        self._base_tools = list(tools or [])
        self._vision_description_model = vision_description_model
        self.tools = list(tools or [])
        if vision_description_model:
            self.tools.append(make_describe_image_tool(vision_description_model))
        self.reflexes = reflexes or []
        self.recursion_limit = recursion_limit
        self._model_name = model
        self.model_name = model
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

    def with_extra_tools(self, tools: list[BaseTool]) -> "AugAgent":
        """A new agent with this one's settings and *tools* added to its own."""
        return AugAgent(tools=[*self._base_tools, *tools], **self._settings)

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
        messages = await _expand_images(messages, vision=not self._vision_description_model)
        logger.debug("llm_call model=%s messages=%d", self._model_name, len(messages))
        response: AIMessage = await self._llm.ainvoke(messages)
        log_token_usage(response)
        return AgentStateUpdate(messages=[*state_changes, response])


def _stamp(content: str | list, now: str) -> str | list:
    """Prepend a timestamp to a message without destroying multimodal content."""
    if isinstance(content, str):
        return f"[{now}] {content}"
    return [{"type": "text", "text": f"[{now}]"}, *content]


# Requests carry at most this many images (Anthropic refuses more than 100); older ones
# drop to their hint.  Compaction usually trims the history well before that.
MAX_INLINE_IMAGES = 50


async def _expand_images(messages: list[AnyMessage], *, vision: bool) -> list[AnyMessage]:
    """Expand the [[img:path|mime]] markers in every user message for the model call.

    With *vision* each marker becomes the image itself plus a hint with its path;
    without, only the hint (the agent then has describe_image).  History keeps the
    markers, and every call expands all of them the same way, re-reading each image
    from its path, so the prompt prefix stays byte-identical and cached from turn to
    turn.

    Only an image AUG stored itself is inlined: one in the workspace file store, of a
    type models accept.  Anything else a marker names (an old upload, a path typed into
    a message) gets just the hint, so one bad image can't fail every later call.
    """
    total = sum(
        len(IMAGE_MARKER.findall(_text(m.content))) for m in messages if isinstance(m, HumanMessage)
    )
    budget = _ImageBudget(skip=max(0, total - MAX_INLINE_IMAGES) if vision else total)
    result = list(messages)
    for i, msg in enumerate(messages):
        if not isinstance(msg, HumanMessage):
            continue
        blocks = await _inline_image_markers(msg.content, budget)
        if blocks is not None:
            result[i] = HumanMessage(content=blocks, id=msg.id)
    return result


class _ImageBudget:
    """Which markers become images: all but the oldest *skip*."""

    def __init__(self, skip: int) -> None:
        self._skip = skip

    def take(self) -> bool:
        if self._skip > 0:
            self._skip -= 1
            return False
        return True


async def _inline_image_markers(content: str | list, budget: _ImageBudget) -> list | None:
    """*content* with its markers expanded, or None when it has none."""
    if isinstance(content, str):
        if not IMAGE_MARKER.search(content):
            return None
        return await _markers_to_blocks(content, budget)
    if isinstance(content, list):
        expanded = []
        changed = False
        for block in content:
            if block.get("type") == "text" and IMAGE_MARKER.search(block["text"]):
                expanded.extend(await _markers_to_blocks(block["text"], budget))
                changed = True
            else:
                expanded.append(block)
        return expanded if changed else None
    return None


async def _markers_to_blocks(text: str, budget: _ImageBudget) -> list:
    """Split a string containing [[img:path|mime]] markers into content blocks."""
    blocks = []
    pos = 0
    for m in IMAGE_MARKER.finditer(text):
        before = text[pos : m.start()].strip()
        if before:
            blocks.append({"type": "text", "text": before})
        path, mime_type = m.group(1), m.group(2)
        if budget.take() and mime_type in MODEL_IMAGE_TYPES:
            data = await asyncio.to_thread(_read_stored_image, Path(path))
            if data is not None:
                encoded = base64.b64encode(data).decode()
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    }
                )
        blocks.append({"type": "text", "text": USER_IMAGE_HINT.format(path=path)})
        pos = m.end()
    after = text[pos:].strip()
    if after:
        blocks.append({"type": "text", "text": after})
    return blocks


def _read_stored_image(path: Path) -> bytes | None:
    """The bytes of *path* if it is a file in the workspace file store, else None."""
    resolved = path.resolve()
    if not resolved.is_relative_to(file_store.root.resolve()) or not resolved.is_file():
        return None
    return resolved.read_bytes()


def _text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


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
