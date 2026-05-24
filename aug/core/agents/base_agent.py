"""Base class for all AUG agents."""

import asyncio
import contextvars
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig, var_child_runnable_config
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import StateSnapshot

from aug.core.events import AgentEvent, ToolEndEvent, ToolStartEvent, parse_event
from aug.core.reflexes import Reflex
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

    The graph is compiled with ``interrupt_after=["call_tools"]`` so that
    _agent_stream in base.py can check for soft interrupts and inject new
    HumanMessages between tool rounds via Command(update=…).

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
    reflexes: list[Reflex] = []
    recursion_limit: int = 25

    def __init__(self) -> None:
        self._compiled_graph = None
        self._subagent_graph = None

    async def astream_events(
        self,
        state: AgentState | Any,
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

    async def aget_state(
        self, config: RunnableConfig, checkpointer: BaseCheckpointSaver
    ) -> StateSnapshot:
        if self._compiled_graph is None:
            self._compiled_graph = self._build(checkpointer)
        return await self._compiled_graph.aget_state(config)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Write values directly to the thread checkpoint without running the graph.

        Uses LangGraph's update_state so reducers (e.g. add_messages) are applied
        correctly.  No LLM call is made.
        """
        if self._compiled_graph is None:
            self._compiled_graph = self._build(checkpointer)
        await self._compiled_graph.aupdate_state(config, values)

    def preprocess(self, state: AgentState) -> AgentStateUpdate:
        """Prepare state before the LLM is called."""
        return AgentStateUpdate()

    @abstractmethod
    async def respond(self, state: AgentState) -> AgentStateUpdate:
        """Call the LLM and return its response (including any tool call requests)."""

    def postprocess(self, state: AgentState) -> AgentStateUpdate:
        """Transform or log the final LLM response before returning to the user."""
        return AgentStateUpdate()

    async def arun(
        self,
        prompt: str,
        *,
        interface: str,
        sender_id: str,
        thread_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent headlessly and yield events.

        Compiles a non-interruptible graph (no checkpointer, no interrupt_after) on
        first call and caches it.  Callers should accumulate ChatModelStreamEvent deltas
        for the final response text.  GraphRecursionError propagates to the caller.

        The subagent graph runs in an isolated asyncio context with
        var_child_runnable_config cleared so its internal tool events do not leak
        into the outer agent's callback chain (which would cause subagent tool calls
        to appear as top-level items in the parent's event stream).
        """
        if self._subagent_graph is None:
            self._subagent_graph = self._build_subagent()
        config: RunnableConfig = {
            "recursion_limit": self.recursion_limit,
            "configurable": {
                "thread_id": thread_id,
                "interface": interface,
                "sender_id": sender_id,
                # Subagents run without a checkpointer, so approval interrupts cannot
                # pause/resume here.  Tools see this and fail honestly instead of
                # silently halting the whole subagent graph.
                "can_approve": False,
            },
        }
        state = AgentState(
            messages=[HumanMessage(content=prompt)],
            thread_id=thread_id,
            interface=interface,
        )

        queue: asyncio.Queue = asyncio.Queue()
        exc_holder: list[BaseException] = []

        async def _collect() -> None:
            try:
                async for raw in self._subagent_graph.astream_events(
                    state, config=config, version="v2"
                ):
                    event = parse_event(raw)
                    if event is not None:
                        await queue.put(event)
            except BaseException as exc:
                exc_holder.append(exc)
            finally:
                await queue.put(None)

        ctx = contextvars.copy_context()
        ctx.run(var_child_runnable_config.set, None)
        task = asyncio.get_running_loop().create_task(_collect(), context=ctx)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if exc_holder:
            raise exc_holder[0]

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state.messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "call_tools"
        return "postprocess"

    def _build_graph(self) -> StateGraph:
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
        return graph

    def _build(self, checkpointer: BaseCheckpointSaver):
        return self._build_graph().compile(
            checkpointer=checkpointer, interrupt_after=["call_tools"]
        )

    def _build_subagent(self):
        return self._build_graph().compile()
