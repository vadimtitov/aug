"""One chat's Codex thread, and what Codex says about it turned into harness events.

A turn is ours from the moment we ask for it: the turn Codex names in its answer (or,
for a compaction, the next one it starts) is that turn, and it ends at its
``turn/completed``.  A turn the thread starts by itself (a subagent's answer arriving
after the turn that spawned it ended) is unattended: its events go to
``on_unsolicited``.  A subagent runs on a thread of its own; each of its turns shows as
one ``subagent`` call, with its work filed under it, as on Claude Code.

A turn always ends: when Codex says so, when the app-server goes away, a few seconds
after a stop Codex never confirms, or once Codex has said nothing for a long while with
nothing in progress.
"""

import asyncio
import logging
import shlex
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from langchain_core.tools import BaseTool

from aug.core.harness.codex.guard import IMAGE_TOOL, PATCH_TOOL, SHELL_TOOL
from aug.core.harness.codex.runtime import CodexRuntime, CodexUnusable
from aug.core.harness.codex.tools import items_text, run_tool, tool_answer
from aug.core.harness.tool_bridge import ToolOutcome
from aug.core.harness.tool_context import Oneshot, ToolContext
from aug.core.harness.types import (
    Address,
    AgentEvent,
    ContextUsage,
    HarnessError,
    HarnessEvent,
    StatusUpdate,
    TextDelta,
    ToolEnded,
    ToolStarted,
    TurnContext,
    TurnEnded,
    Usage,
)
from aug.core.prompts import HARNESS_COMPACTED_STATUS, HARNESS_TOOL_STOPPED_RESULT
from aug.utils.codex_app_server import AppServerClosed, CodexRpcError, JsonObject
from aug.utils.hushed import unhushed

logger = logging.getLogger(__name__)

# How long a stopped turn waits for Codex to confirm before it ends without it.
INTERRUPT_SECONDS = 10
# How long a turn may hear nothing from Codex, with nothing in progress, before it ends.
SILENT_SECONDS = 15 * 60
# How often a turn's watchdog looks.
_WATCH_SECONDS = 1
# The names AUG's events give Codex's tools the guard doesn't see.
SEARCH_TOOL = "web_search"
SUBAGENT_TOOL = "subagent"
# Codex's error kinds (TurnError.codexErrorInfo) onto AUG's.
_UNAVAILABLE = frozenset(
    {
        "serverOverloaded",
        "internalServerError",
        "httpConnectionFailed",
        "responseStreamConnectionFailed",
        "responseStreamDisconnected",
        "responseTooManyFailedAttempts",
        "policyCapacityUnavailable",
        "flexUnavailable",
    }
)

# What Codex hears for an AUG tool call a stop cut off, or that came too late to run.
_STOPPED = ToolOutcome(blocks=(HARNESS_TOOL_STOPPED_RESULT,), error=True)

# Receives the events of a turn the thread started by itself.
UnsolicitedHandler = Callable[[AsyncIterator[HarnessEvent]], Awaitable[None]]


@dataclass
class _Turn:
    ctx: ToolContext  # emits into this turn's events
    unsolicited: bool
    id: str | None = None  # Codex's id, once it has named the turn
    events: asyncio.Queue[HarnessEvent | None] = field(default_factory=asyncio.Queue)
    error: HarnessError | None = None
    cancelled: bool = False
    ended: bool = False
    # The thread's tokens when it began; None until Codex has first counted them here
    # (a thread resumed on a new process).
    total_at_start: Usage | None = None
    after_text: bool = False  # the last thing the agent did was say something
    heard_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    open_items: set[str] = field(default_factory=set)  # its tool calls in progress
    watchdog: asyncio.Task[None] | None = None


@dataclass
class _Child:
    """A subagent's thread, and the ``subagent`` call its current turn shows as."""

    call_id: str
    parent_id: str | None  # the call it is filed under: a subagent's own subagent
    running: bool = True
    reply: str | None = None  # what its turn last said
    turns: int = 1


class ChatThread:
    """One chat's Codex thread, bound to it by ``bind()``."""

    def __init__(
        self,
        context_id: str,
        runtime: CodexRuntime,
        *,
        tools: Callable[[], Mapping[str, BaseTool]],
        reply_to: Address,
        on_unsolicited: UnsolicitedHandler,
        oneshot: Oneshot | None = None,
    ) -> None:
        """*tools* gives the profile's tools by name as they are now: a call runs the
        current tool.  *oneshot*: for the tools of a turn Codex starts by itself."""
        self.context_id = context_id
        self.reply_to = reply_to
        self.thread_id: str | None = None
        self.last_used = time.monotonic()
        self._runtime = runtime
        self._tools = tools
        self._on_unsolicited = on_unsolicited
        self._oneshot = oneshot
        self._turn: _Turn | None = None
        self._unnamed: _Turn | None = None  # asked for; Codex has not named it yet
        self._children: dict[str, _Child] = {}  # by the subagent's thread id
        self._streamed: set[str] = set()  # agent messages whose text came as deltas
        self._tool_calls: set[asyncio.Task[JsonObject]] = set()
        self._tasks: set[asyncio.Task[Any]] = set()  # deliveries and interrupts in flight
        self._total: Usage | None = None  # the thread's tokens so far, as Codex counts them
        self._context: ContextUsage | None = None
        # Turns ended here while Codex may still run them: their late events are ignored.
        self._stale: set[str] = set()

    def bind(self, thread_id: str) -> None:
        """Hear about Codex's thread *thread_id*: the one this chat is on now.  Raises
        CodexUnusable, changing nothing, if it is another chat's."""
        if not self._runtime.listen(thread_id, self):
            raise CodexUnusable(f"codex thread {thread_id} is another chat's")
        if self.thread_id is not None and self.thread_id != thread_id:
            self._runtime.forget(self.thread_id, self)
        self.thread_id = thread_id
        self.last_used = time.monotonic()

    def close(self) -> None:
        """Stop hearing about the thread and its subagents."""
        for thread_id in (self.thread_id, *self._children):
            if thread_id is not None:
                self._runtime.forget(thread_id, self)
        self._children.clear()

    @property
    def background_tasks(self) -> int:
        """Subagents still running."""
        return sum(child.running for child in self._children.values())

    @property
    def busy(self) -> bool:
        """A turn, a tool call or a subagent is running."""
        return self._turn is not None or bool(self._tool_calls) or self.background_tasks > 0

    @property
    def context(self) -> ContextUsage | None:
        """How full the context is as of the last model call; None before the first,
        or after a compaction until the next."""
        return self._context

    @property
    def turn_open(self) -> bool:
        return self._turn is not None

    async def start(
        self, items: list[JsonObject], ctx: ToolContext, params: JsonObject
    ) -> AsyncIterator[HarnessEvent]:
        """Start a turn with *items* (Codex user input); returns its events, ending
        with TurnEnded.  *params* go into turn/start as they are (model, effort).
        Once this returns, steer() reaches the turn."""
        assert self.thread_id is not None, "bind() first"
        turn = self._ask(ctx)
        try:
            result = await self._runtime.request(
                "turn/start", {"threadId": self.thread_id, "input": items, **params}
            )
        except asyncio.CancelledError:
            self._abandon(turn)
            raise
        except Exception as e:
            self._refused(turn, "turn/start", e)
        else:
            if turn_id := (result.get("turn") or {}).get("id"):
                self._adopt(turn, str(turn_id), answer=True)
        return self._events(turn)

    async def compact(self, ctx: ToolContext) -> AsyncIterator[HarnessEvent]:
        """Compact the thread; returns the compaction's turn as events."""
        assert self.thread_id is not None, "bind() first"
        turn = self._ask(ctx)
        try:
            await self._runtime.request("thread/compact/start", {"threadId": self.thread_id})
        except asyncio.CancelledError:
            self._abandon(turn)
            raise
        except Exception as e:
            self._refused(turn, "thread/compact/start", e)
        return self._events(turn)

    async def steer(self, items: list[JsonObject]) -> bool:
        """Add *items* to the attended turn in flight; False if there is none."""
        turn = self._turn
        if turn is None or turn.unsolicited or turn.cancelled or turn.id is None:
            return False
        try:
            await self._runtime.request(
                "turn/steer",
                {"threadId": self.thread_id, "expectedTurnId": turn.id, "input": items},
            )
        except (CodexRpcError, AppServerClosed, TimeoutError) as e:
            logger.info("codex steer refused context=%s: %s", self.context_id, e)
            return False
        return True

    async def interrupt(self) -> bool:
        """Stop the turn in flight and the AUG tools it is running.  It ends with
        TurnEnded(cancelled=True): when Codex says so, else after INTERRUPT_SECONDS.
        A turn Codex has not named yet is interrupted as soon as it does."""
        turn = self._turn
        if turn is None:
            return False
        await self._stop(turn)
        return True

    # -- ThreadListener ------------------------------------------------------------------

    def on_notification(self, method: str, params: JsonObject) -> None:
        thread_id = str(params.get("threadId"))
        child = self._children.get(thread_id)
        if thread_id != self.thread_id and child is None:
            return
        about = params.get("turnId")
        if child is None and about is not None and method != "thread/tokenUsage/updated":
            if not self._is_ours(about):
                return  # a turn ended here, whose late word nobody hears
        if self._turn is not None:
            self._turn.heard_at = time.monotonic()
        match method:
            case "turn/started" if child is None:
                self._on_turn_started(params.get("turn") or {})
            case "turn/started":
                self._child_turn_started(thread_id)
            case "turn/completed" if child is None:
                self._on_turn_completed(params.get("turn") or {})
            case "turn/completed":
                self._child_turn_done(thread_id)
            case "thread/closed" if child is not None:
                self._child_turn_done(thread_id)
                self._children.pop(thread_id, None)
                self._runtime.forget(thread_id, self)
            case "item/started" | "item/completed":
                self._on_item(params.get("item") or {}, child, method == "item/started")
            case "item/agentMessage/delta":
                self._on_text(str(params.get("itemId")), str(params.get("delta") or ""), child)
            case "thread/tokenUsage/updated" if child is None:
                self._on_usage(params.get("tokenUsage") or {})
            case "error" if child is None and not params.get("willRetry"):
                if self._turn is not None:
                    self._turn.error = _turn_error(params.get("error"), self._runtime)

    async def on_tool_call(self, params: JsonObject) -> JsonObject:
        """Run the AUG tool Codex asked for, with the turn's ToolContext (a subagent's
        between turns, with the chat's).  A /stop cancels it, and a call for a turn
        that is stopped or over never runs: Codex is answered that it did not finish."""
        turn = self._turn
        name = str(params.get("tool"))
        from_child = str(params.get("threadId")) in self._children
        if (turn is not None and turn.cancelled) or (
            not from_child and not self._is_ours(params.get("turnId"))
        ):
            logger.info("codex called %s for a turn that is over context=%s", name, self.context_id)
            return tool_answer(_STOPPED)
        ctx = turn.ctx if turn is not None else self._unattended()
        call = asyncio.create_task(
            run_tool(self._tools(), name, params.get("arguments"), str(params.get("callId")), ctx)
        )
        self._tool_calls.add(call)
        try:
            await asyncio.wait({call})
        finally:
            self._tool_calls.discard(call)
            call.cancel()  # a no-op once it is done; else nobody waits for its answer
        if call.cancelled():
            return tool_answer(_STOPPED)
        return call.result()

    def on_closed(self) -> None:
        self._cancel_tool_calls()
        for thread_id in self._children:
            self._runtime.forget(thread_id, self)
        self._children.clear()
        self._stale.clear()
        self._unnamed = None
        if self._turn is not None:
            self._turn.error = self._turn.error or HarnessError(kind="unavailable")
            self._end(self._turn)

    # -- turns ---------------------------------------------------------------------------

    def _ask(self, ctx: ToolContext) -> _Turn:
        """Open the turn we are about to ask Codex for.  A turn the thread started by
        itself just before ends here: Codex joins our input to it, and its remaining
        events go to the user who is here now."""
        if self._turn is not None and self._turn.unsolicited:
            # Its late word is ignored, unless Codex names it as ours (see _adopt).
            self._end(self._turn, codex_running=True)
        self.last_used = time.monotonic()
        turn = self._open(ctx, unsolicited=False)
        self._unnamed = turn
        return turn

    def _open(self, ctx: ToolContext, *, unsolicited: bool) -> _Turn:
        turn = _Turn(
            ctx=replace(ctx, _emit=self._emit),
            unsolicited=unsolicited,
            total_at_start=self._total,
        )
        turn.watchdog = asyncio.create_task(self._watch(turn), name=f"codex-{self.context_id}")
        self._turn = turn
        return turn

    def _unattended(self) -> ToolContext:
        """The context of a turn Codex starts by itself, and of a tool a subagent calls
        between the chat's turns."""
        return ToolContext(
            context_id=self.context_id,
            turn=TurnContext(reply_to=self.reply_to, attended=False),
            _oneshot=self._oneshot,
        )

    def _is_ours(self, turn_id: Any) -> bool:
        """Whether what Codex says about its turn *turn_id* (None: no turn named) is
        about the turn open here."""
        turn = self._turn
        if turn is None:
            return False
        if turn_id is None:
            return True
        return str(turn_id) not in self._stale and turn.id in (None, str(turn_id))

    def _abandon(self, turn: _Turn) -> None:
        """Nobody waits for *turn* any more: it ends here and is interrupted, now if
        Codex has named it, else once it does (it stays the turn we asked for)."""
        turn.cancelled = True
        self._end(turn, codex_running=True)
        if turn.id is not None:
            self._spawn(self._interrupt_id(turn.id))

    def _refused(self, turn: _Turn, method: str, error: Exception) -> None:
        """Codex would not start *turn*: it ends now.  (A request it never answered
        restarted the app-server, so nothing starts later either.)"""
        if self._unnamed is turn:
            self._unnamed = None
        if isinstance(error, CodexRpcError | AppServerClosed | TimeoutError):
            logger.warning("codex %s failed context=%s: %s", method, self.context_id, error)
        else:
            logger.exception("codex %s failed context=%s", method, self.context_id)
        turn.error = turn.error or _rpc_error(error)
        self._end(turn)

    def _adopt(self, turn: _Turn, turn_id: str, *, answer: bool = False) -> None:
        """*turn* is Codex's turn *turn_id*: as the first ``turn/started`` after we
        asked has it, or as *answer* to turn/start says, which wins.  One stopped
        before then is interrupted now; one already ended here is interrupted and its
        events ignored."""
        if self._unnamed is turn:
            self._unnamed = None
        if turn.id == turn_id:
            return
        if turn.id is not None:
            if not answer:
                return
            # That turn/started was a turn Codex started by itself as we asked for
            # ours: Codex runs it on, but nobody here hears it.
            logger.warning("codex turn %s started alongside ours, ignored", turn.id)
            self._stale.add(turn.id)
        self._stale.discard(turn_id)
        turn.id = turn_id
        if turn.ended:
            self._stale.add(turn_id)
            self._spawn(self._interrupt_id(turn_id))
        elif turn.cancelled:
            self._spawn(self._send_interrupt(turn))

    def _on_turn_started(self, raw: JsonObject) -> None:
        turn_id = str(raw.get("id") or "")
        if not turn_id or turn_id in self._stale:
            return
        if self._unnamed is not None:
            self._adopt(self._unnamed, turn_id)
        elif self._turn is None:
            logger.info("codex started a turn by itself context=%s", self.context_id)
            turn = self._open(self._unattended(), unsolicited=True)
            turn.id = turn_id
            self._spawn(self._on_unsolicited(self._events(turn)))

    def _on_turn_completed(self, raw: JsonObject) -> None:
        turn_id = str(raw.get("id") or "")
        if turn_id in self._stale:
            self._stale.discard(turn_id)
            return
        turn = self._turn
        if turn is None or turn.id != turn_id:
            return
        status = raw.get("status")
        if status == "interrupted":
            turn.cancelled = True
        elif status == "failed" and not turn.cancelled:
            turn.error = turn.error or _turn_error(raw.get("error"), self._runtime)
        self._end(turn)

    async def _stop(self, turn: _Turn) -> None:
        """Stop *turn* and the AUG tools running.  It ends with TurnEnded(cancelled=True):
        when Codex says so, else after INTERRUPT_SECONDS."""
        if turn.ended:
            return
        turn.cancelled = True
        if turn.stopped_at is None:
            turn.stopped_at = time.monotonic()
        turn.ctx.stopped.set()
        self._cancel_tool_calls()
        if turn.id is not None:
            await self._send_interrupt(turn)

    async def _send_interrupt(self, turn: _Turn) -> None:
        """Interrupt *turn*; one Codex won't interrupt ends here all the same."""
        assert turn.id is not None
        if not await self._interrupt_id(turn.id):
            self._end(turn, codex_running=True)

    async def _interrupt_id(self, turn_id: str) -> bool:
        """Ask Codex to interrupt its turn *turn_id*; False if it would not."""
        try:
            async with asyncio.timeout(INTERRUPT_SECONDS):
                await self._runtime.request(
                    "turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id}
                )
        except (CodexRpcError, AppServerClosed, TimeoutError) as e:
            logger.warning("codex interrupt failed context=%s: %s", self.context_id, e)
            return False
        return True

    async def _watch(self, turn: _Turn) -> None:
        """End *turn* if Codex never confirms its stop, or goes silent while nothing
        is in progress."""
        while not turn.ended:
            await asyncio.sleep(_WATCH_SECONDS)
            now = time.monotonic()
            if turn.stopped_at is not None and now - turn.stopped_at >= INTERRUPT_SECONDS:
                logger.warning("codex never ended a stopped turn context=%s", self.context_id)
                self._end(turn, codex_running=True)
            elif now - turn.heard_at >= SILENT_SECONDS and not self._working(turn):
                logger.warning("codex went silent mid-turn context=%s", self.context_id)
                turn.error = HarnessError(kind="unavailable")
                self._cancel_tool_calls()
                if turn.id is not None:
                    self._spawn(self._interrupt_id(turn.id))
                self._end(turn, codex_running=True)

    def _working(self, turn: _Turn) -> bool:
        """Something that may rightly take long is in progress: a command, a tool call
        (an approval can wait half an hour), a subagent."""
        return bool(turn.open_items or self._tool_calls) or self.background_tasks > 0

    def _cancel_tool_calls(self) -> None:
        for call in self._tool_calls:
            call.cancel()

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task: asyncio.Task[Any] = asyncio.create_task(coro, name=f"codex-{self.context_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _end(self, turn: _Turn, *, codex_running: bool = False) -> None:
        """Close *turn*'s stream.  *codex_running*: Codex may still be running it, so
        what it says about it later is ignored."""
        if turn.ended:
            return
        turn.ended = True
        if turn.watchdog is not None and turn.watchdog is not asyncio.current_task():
            turn.watchdog.cancel()
        if codex_running and turn.id is not None:
            self._stale.add(turn.id)
        self._streamed.clear()
        turn.events.put_nowait(
            TurnEnded(
                error=None if turn.cancelled else turn.error,
                cancelled=turn.cancelled,
                usage=_minus(self._total, turn.total_at_start)
                if self._total is not None and turn.total_at_start is not None
                else Usage(),
                context=self._context,
            )
        )
        turn.events.put_nowait(None)
        if self._turn is turn:
            self._turn = None

    async def _events(self, turn: _Turn) -> AsyncIterator[HarnessEvent]:
        """*turn*'s events, up to its TurnEnded.  One whose reader stops early is
        interrupted: nobody would see the rest."""
        try:
            while (event := await turn.events.get()) is not None:
                yield event
        finally:
            if not turn.ended:
                self._spawn(self._stop(turn))

    def _put(self, event: HarnessEvent) -> None:
        if self._turn is not None:
            self._turn.events.put_nowait(event)

    async def _emit(self, event: AgentEvent) -> None:
        self._put(event)

    # -- what Codex says -----------------------------------------------------------------

    def _on_text(self, item_id: str, delta: str, child: _Child | None) -> None:
        if not delta:
            return
        if child is None:
            self._streamed.add(item_id)
        self._put(TextDelta(text=delta, parent_id=child.call_id if child is not None else None))

    def _on_item(self, item: JsonObject, child: _Child | None, started: bool) -> None:
        """What Codex (or a subagent, *child*) is doing; *started*: it begins, else it
        is done."""
        kind = item.get("type")
        item_id = str(item.get("id", ""))
        parent = child.call_id if child is not None else None
        turn = self._turn
        if kind == "agentMessage":
            self._on_message(item, item_id, child, started)
        elif kind == "subAgentActivity":
            self._on_subagent(item, parent, started)
        elif kind == "contextCompaction" and not started:
            self._context = None
            self._put(StatusUpdate(text=HARNESS_COMPACTED_STATUS))
        elif (tool := _tool_item(item)) is not None:
            name, args, output, error = tool
            if turn is not None and child is None:
                turn.after_text = False
                if started:
                    turn.open_items.add(item_id)
                else:
                    turn.open_items.discard(item_id)
            if started:
                self._put(ToolStarted(id=item_id, name=name, args=args, parent_id=parent))
            else:
                self._put(
                    ToolEnded(id=item_id, name=name, output=output, error=error, parent_id=parent)
                )

    def _on_message(
        self, item: JsonObject, item_id: str, child: _Child | None, started: bool
    ) -> None:
        """The agent's own words stream as they come; a subagent's last words are its
        call's output."""
        turn = self._turn
        if child is not None:
            if not started:
                child.reply = str(item.get("text") or "")
            return
        if started and turn is not None and turn.after_text:
            self._put(TextDelta(text="\n\n"))
        elif not started:
            if item_id not in self._streamed and item.get("text"):
                self._put(TextDelta(text=str(item["text"])))
            self._streamed.discard(item_id)
            if turn is not None:
                turn.after_text = True

    def _on_subagent(self, item: JsonObject, parent: str | None, started: bool) -> None:
        """Codex spawning a subagent, or giving it more to do, opens its call; its
        finishing or being stopped ends it."""
        thread_id = str(item.get("agentThreadId", ""))
        kind = item.get("kind")
        if not thread_id:
            return
        if kind in ("started", "interacted") and started:
            self._child_turn_started(thread_id, parent=parent, task=item.get("agentPath"))
        elif kind in ("completed", "interrupted") and not started:
            self._child_turn_done(thread_id, interrupted=kind == "interrupted")

    def _child_turn_started(
        self, thread_id: str, *, parent: str | None = None, task: Any = None
    ) -> None:
        child = self._children.get(thread_id)
        if child is not None and child.running:
            return
        if child is None:
            if not self._runtime.listen(thread_id, self):
                return  # another chat's thread
            child = _Child(call_id=thread_id, parent_id=parent)
            self._children[thread_id] = child
        else:
            child.turns += 1
            child.call_id = f"{thread_id}:{child.turns}"
            child.running = True
        self._put(
            ToolStarted(
                id=child.call_id,
                name=SUBAGENT_TOOL,
                args={"task": str(task or "")},
                parent_id=child.parent_id,
            )
        )

    def _child_turn_done(self, thread_id: str, *, interrupted: bool = False) -> None:
        child = self._children.get(thread_id)
        if child is None or not child.running:
            return
        child.running = False
        self._put(
            ToolEnded(
                id=child.call_id,
                name=SUBAGENT_TOOL,
                output=child.reply,
                error=interrupted,
                parent_id=child.parent_id,
            )
        )
        child.reply = None

    def _on_usage(self, raw: JsonObject) -> None:
        total, last = raw.get("total") or {}, raw.get("last") or {}
        self._total = _usage(total)
        turn = self._turn
        if turn is not None and turn.total_at_start is None:
            # The first count since the thread opened here: before this model call.
            turn.total_at_start = _minus(self._total, _usage(last))
        window = raw.get("modelContextWindow")
        used = int(last.get("inputTokens") or 0) + int(last.get("outputTokens") or 0)
        if window and used:
            self._context = ContextUsage(used=used, window=int(window))


def original_command(command: str) -> str:
    """The command the agent wrote, from what Codex ran: its shell around the guard's
    hushed wrapper around the command (``/bin/bash -c '<hushed argv>'``)."""
    try:
        outer = shlex.split(command)
    except ValueError:
        return command
    line = outer[-1] if len(outer) >= 3 and outer[-2] in ("-c", "-lc") else command
    while (inner := unhushed(line)) is not None:
        line = inner
    return line


def _tool_item(item: JsonObject) -> tuple[str, dict, str | None, bool] | None:
    """A tool call item as (name, args, output, error); None for anything else."""
    status = item.get("status")
    failed = status in ("failed", "declined")
    match item.get("type"):
        case "commandExecution":
            exit_code = item.get("exitCode")
            return (
                SHELL_TOOL,
                {"command": original_command(str(item.get("command", "")))},
                item.get("aggregatedOutput"),
                failed or (exit_code is not None and exit_code != 0),
            )
        case "fileChange":
            paths = [str(c.get("path")) for c in item.get("changes") or [] if isinstance(c, dict)]
            return PATCH_TOOL, {"files": paths}, "\n".join(paths), failed
        case "dynamicToolCall":
            args = item.get("arguments")
            return (
                str(item.get("tool")),
                args if isinstance(args, dict) else {},
                items_text(item.get("contentItems")),
                failed or item.get("success") is False,
            )
        case "mcpToolCall":
            args = item.get("arguments")
            error = item.get("error")
            result = item.get("result") or {}
            output = (
                error.get("message")
                if isinstance(error, dict)
                else "\n".join(
                    str(c.get("text", ""))
                    for c in result.get("content") or []
                    if isinstance(c, dict)
                )
            )
            return (
                f"{item.get('server')}__{item.get('tool')}",
                args if isinstance(args, dict) else {},
                output,
                failed or error is not None,
            )
        case "webSearch":
            return SEARCH_TOOL, {"query": item.get("query", "")}, None, failed
        case "imageView":
            return IMAGE_TOOL, {"path": item.get("path", "")}, None, failed
        case "collabAgentToolCall":
            return str(item.get("tool")), {"prompt": item.get("prompt") or ""}, None, failed
    return None


def _turn_error(raw: Any, runtime: CodexRuntime) -> HarnessError:
    """Codex's TurnError onto AUG's error kinds."""
    info = raw.get("codexErrorInfo") if isinstance(raw, dict) else None
    key = next(iter(info), None) if isinstance(info, dict) else info
    # The HTTP errors carry the status: {"httpConnectionFailed": {"httpStatusCode": 401}}.
    detail = info.get(key) if isinstance(info, dict) else None
    status = detail.get("httpStatusCode") if isinstance(detail, dict) else None
    message = raw.get("message") if isinstance(raw, dict) else raw
    logger.warning("codex turn error %s status=%s: %.200s", key, status, message)
    if key == "contextWindowExceeded":
        return HarnessError(kind="context_full")
    if key in ("usageLimitExceeded", "rateLimitExceeded"):
        return HarnessError(kind="rate_limited", retry_at=runtime.retry_at())
    if key == "unauthorized" or status in (401, 403):
        return HarnessError(kind="login", login_with="codex")
    if status == 429:
        return HarnessError(kind="rate_limited", retry_at=runtime.retry_at())
    if key in _UNAVAILABLE:
        return HarnessError(kind="unavailable")
    if key == "sessionBudgetExceeded":
        return HarnessError(kind="step_limit")
    return HarnessError(kind="internal")


def _rpc_error(error: Exception) -> HarnessError:
    if isinstance(error, AppServerClosed | TimeoutError):
        return HarnessError(kind="unavailable")
    if isinstance(error, CodexRpcError) and "authentication required" in error.message.lower():
        return HarnessError(kind="login", login_with="codex")
    return HarnessError(kind="internal")


def _usage(raw: JsonObject) -> Usage:
    cached = int(raw.get("cachedInputTokens") or 0)
    return Usage(
        input_tokens=int(raw.get("inputTokens") or 0) - cached,
        output_tokens=int(raw.get("outputTokens") or 0),
        cache_read_tokens=cached,
        cache_write_tokens=int(raw.get("cacheWriteInputTokens") or 0),
    )


def _minus(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=max(0, a.input_tokens - b.input_tokens),
        output_tokens=max(0, a.output_tokens - b.output_tokens),
        cache_read_tokens=max(0, a.cache_read_tokens - b.cache_read_tokens),
        cache_write_tokens=max(0, a.cache_write_tokens - b.cache_write_tokens),
    )
