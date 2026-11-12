"""LangGraphHarness: today's agent graph behind the Harness interface.

Owns everything LangGraph-specific about a turn: the stream loop across
``interrupt_after=["call_tools"]`` pauses, mid-run injection, the stop flag, approval
interrupts and their resume, reflexes, compaction, and mapping model/graph exceptions
onto HarnessError kinds.  Interfaces never see a LangGraph type from here except the
parsed graph events, which PR 2 replaces with harness events.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import openai
import psycopg
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from aug.core.agents.base_agent import BaseAgent
from aug.core.compaction import compact_thread
from aug.core.events import StatusEvent
from aug.core.harness.base import Harness
from aug.core.harness.types import (
    FilePart,
    HarnessError,
    HarnessEvent,
    Message,
    ModelTier,
    Sender,
    TextPart,
    TurnContext,
    TurnEnded,
)
from aug.core.llm import build_chat_model
from aug.core.prompts import MID_RUN_INJECTION_PREFIX
from aug.core.reflexes import Reflex, run_reflexes
from aug.core.run import AGENT_RUN_CONFIG_KEY, AgentRun, MessageContent, run_registry
from aug.core.state import AgentState
from aug.core.tools.approval import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)


class LangGraphHarness(Harness):
    """One agent version's compiled graph, sharing the app's checkpointer."""

    def __init__(self, agent: BaseAgent, checkpointer: BaseCheckpointSaver) -> None:
        self.agent = agent
        self._checkpointer = checkpointer
        self._opened: dict[str, AgentRun] = {}

    def _run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        graph_input = AgentState(
            messages=[HumanMessage(content=_render(message))],
            thread_id=context_id,
            interface=turn.reply_to.interface,
        )
        # Reflexes answer what the user just said; a follow-up turn's messages were
        # injections (their own reflexes already ran), so it fires none.
        return self._stream(graph_input, message, context_id, turn, reflexes=not turn.follow_up)

    async def resume(
        self, decision: ApprovalDecision, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        """Resume a turn paused on an approval, with the user's *decision*.  Same
        contract as run(): never raises, ends with TurnEnded."""
        inner = self._stream(Command(resume=decision), None, context_id, turn, reflexes=False)
        async for event in self._turn(None, context_id, turn, inner):
            yield event

    async def pending_approval(self, context_id: str) -> ApprovalRequest | None:
        """The approval *context_id* is paused on, if any."""
        try:
            state = await self.agent.aget_state(_config(context_id), self._checkpointer)
            for interrupt_obj in state.interrupts:
                if isinstance(interrupt_obj.value, ApprovalRequest):
                    return interrupt_obj.value
        except Exception:
            logger.warning("approval_state_check_failed context=%s", context_id, exc_info=True)
        return None

    async def add_context(self, text: str, context_id: str) -> None:
        """Append *text* to an idle session's history without running a turn."""
        await self.agent.aupdate_state(
            _config(context_id), {"messages": [HumanMessage(content=text)]}, self._checkpointer
        )

    async def inject(self, message: Message, context_id: str) -> bool:
        if not run_registry.accepting(context_id):
            return False
        run_registry.get(context_id).inject_message(_render(message))  # type: ignore[union-attr]
        return True

    async def cancel(self, context_id: str) -> bool:
        run = run_registry.get(context_id)
        if run is None or not run.active:
            return False
        logger.info("cancel context=%s run=%s", context_id, run.id)
        run.request_stop()
        return True

    async def compact(self, context_id: str) -> bool:
        return await compact_thread(self.agent, context_id, self._checkpointer)

    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG) -> str:
        if not self.agent.model_name:
            raise ValueError(f"Agent '{type(self.agent).__name__}' has no model for oneshot()")
        response = await build_chat_model(self.agent.model_name, temperature=0.0).ainvoke(prompt)
        return response.text

    def _open_turn(self, context_id: str, turn: TurnContext) -> None:
        existing = run_registry.get(context_id)
        if existing is not None and existing.active:
            existing.request_stop()  # the run lock should rule this out; never leave two
        run = AgentRun(accepts_injection=turn.attended)
        run_registry.set(context_id, run)
        self._opened[context_id] = run

    def _close_turn(self, context_id: str) -> None:
        run = self._opened.pop(context_id, None)
        if run is None:
            return
        run.active = False
        if run_registry.get(context_id) is run:  # never drop a turn this one didn't open
            run_registry.pop(context_id)

    async def _stream(
        self,
        graph_input: AgentState | Command | None,
        message: Message | None,
        context_id: str,
        turn: TurnContext,
        *,
        reflexes: bool,
    ) -> AsyncIterator[HarnessEvent]:
        """Stream the graph across tool-round pauses, injecting queued messages."""
        run = run_registry.get(context_id)
        assert run is not None, "_open_turn registers the run before the stream starts"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": context_id,
                "interface": turn.reply_to.interface,
                "sender_id": turn.reply_to.chat_id,
                AGENT_RUN_CONFIG_KEY: run,
            }
        }
        finished = False
        displays: list[str] = []  # reflex results to show the user, as they land
        reflex_task: asyncio.Task[None] | None = None
        if reflexes and message is not None and self.agent.reflexes:
            reflex_task = asyncio.create_task(
                self._run_reflexes(self.agent.reflexes, message.text, context_id, run, displays)
            )
        try:
            while True:
                async for event in self.agent.astream_events(
                    graph_input, config, self._checkpointer
                ):
                    if run.user_requested_stop.is_set():
                        break
                    yield event
                    while displays:
                        yield StatusEvent(text=displays.pop(0))
                if run.user_requested_stop.is_set():
                    finished = True
                    break
                state = await self.agent.aget_state(config, self._checkpointer)
                if not state.next:
                    finished = True
                    break
                approval = _approval_in(state.interrupts)
                if approval is not None or state.interrupts:
                    yield TurnEnded(approval=approval)
                    return
                try:
                    injection = run.pending_agent_injection.get_nowait()
                except asyncio.QueueEmpty:
                    graph_input = None  # resume where the graph paused
                    continue
                logger.info("soft_interrupt context=%s run=%s", context_id, run.id)
                graph_input = Command(
                    update={"messages": [HumanMessage(content=_frame_injection(injection))]}
                )
        except Exception as e:
            yield TurnEnded(error=_harness_error(e))
            return
        finally:
            # A turn that didn't finish (error, approval pause, consumer gone) drops its
            # reflexes; one that did waits for them below so their inject is counted.
            if reflex_task is not None and not reflex_task.done() and not finished:
                reflex_task.cancel()

        if reflex_task is not None:
            await asyncio.gather(reflex_task, return_exceptions=True)
        for text in displays:
            yield StatusEvent(text=text)
        # Stop taking injections before collecting what's queued: from here on a new
        # message gets a turn of its own instead of landing in a queue nobody reads.
        run.active = False
        sender = message.sender if message is not None else Sender(turn.reply_to.interface, "")
        unconsumed = []
        while True:
            try:
                leftover = run.pending_agent_injection.get_nowait()
            except asyncio.QueueEmpty:
                break
            unconsumed.append(Message(parts=_parts(leftover), sender=sender))
        yield TurnEnded(unconsumed=unconsumed, cancelled=run.user_requested_stop.is_set())

    async def _run_reflexes(
        self,
        reflexes: list[Reflex],
        query: str,
        context_id: str,
        run: AgentRun,
        displays: list[str],
    ) -> None:
        """Run reflexes alongside the turn; each result is injected as soon as it lands,
        so the agent can pick it up at its next tool-round pause, and its display text
        is queued for the turn's event stream."""
        history = await self._recent_history(context_id)
        for result in await run_reflexes(reflexes, query, history):
            run.inject_message(result.inject)
            if result.display:
                displays.append(result.display)

    async def _recent_history(self, context_id: str, n: int = 3) -> list[MessageContent]:
        """The last *n* user/assistant messages, as labelled strings, for reflexes."""
        try:
            snapshot = await self.agent.aget_state(_config(context_id), self._checkpointer)
        except Exception:
            logger.warning("history_fetch_failed context=%s", context_id, exc_info=True)
            return []
        messages = (snapshot.values or {}).get("messages", [])
        result: list[MessageContent] = []
        for msg in [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-n:]:
            prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
            if isinstance(msg.content, str):
                result.append(f"{prefix}: {msg.content}")
            elif isinstance(msg.content, list):
                texts = [
                    b["text"]
                    for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if texts:
                    result.append(f"{prefix}: {' '.join(texts)}")
        return result


def _config(context_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": context_id}}


def _render(message: Message) -> str:
    """The text the model sees for *message*.

    Images become ``[[img:path|mime]]`` markers, which the agent expands into image
    blocks (or describes, for models without vision) when it calls the model.
    """
    blocks = []
    for part in message.parts:
        match part:
            case TextPart(text=text):
                blocks.append(text)
            case FilePart(path=path, media_type=media_type) if media_type.startswith("image/"):
                blocks.append(f"[[img:{path}|{media_type}]]")
            case FilePart(path=path, media_type=media_type, filename=filename):
                blocks.append(f"[File uploaded: {filename} ({media_type}) — saved to {path}]")
    return "\n\n".join(blocks)


def _parts(content: MessageContent) -> list[TextPart | FilePart]:
    if isinstance(content, str):
        return [TextPart(text=content)]
    return [TextPart(text=b["text"]) for b in content if b.get("type") == "text"]


def _frame_injection(content: MessageContent) -> MessageContent:
    """Prepend MID_RUN_INJECTION_PREFIX so the model knows not to abandon its task."""
    if isinstance(content, str):
        return MID_RUN_INJECTION_PREFIX + content
    return [{"type": "text", "text": MID_RUN_INJECTION_PREFIX}, *content]


def _approval_in(interrupts) -> ApprovalRequest | None:
    for interrupt_obj in interrupts:
        if isinstance(interrupt_obj.value, ApprovalRequest):
            return interrupt_obj.value
    return None


def _harness_error(e: Exception) -> HarnessError:
    """Map a model-client or graph exception onto a HarnessError kind."""
    if isinstance(e, ContextOverflowError) or (
        isinstance(e, openai.BadRequestError)
        and getattr(e, "code", None) == "context_length_exceeded"
    ):
        logger.warning("context window full: %s", e)
        return HarnessError(kind="context_full")
    if isinstance(e, openai.RateLimitError):
        logger.warning("LLM rate limit hit: %s", e)
        return HarnessError(kind="rate_limited")
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
        logger.warning("LLM authentication failed: %s", e)
        return HarnessError(kind="auth")
    if isinstance(
        e, (openai.InternalServerError, openai.APIConnectionError, psycopg.OperationalError)
    ):
        logger.warning("dependency unavailable: %s: %s", type(e).__name__, e)
        return HarnessError(kind="unavailable")
    if isinstance(e, GraphRecursionError):
        logger.warning("agent hit recursion limit")
        return HarnessError(kind="step_limit")
    logger.exception("Unhandled error in agent turn: %s: %s", type(e).__name__, e)
    return HarnessError(kind="internal")
