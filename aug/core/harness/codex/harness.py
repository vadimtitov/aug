"""CodexHarness: a profile run on Codex (``codex app-server``), signed in with ChatGPT.

Every chat has its own Codex thread, all of them on the one app-server process the
Codex profiles share (CodexRuntime).  What AUG keeps about a chat's thread is a record:
the thread's id, and the prompt and address it started with.  Codex keeps the thread
itself (its rollout file in CODEX_HOME), so a thread resumes after a restart as it was.
Threads idle past MAX_OPEN_THREADS are let go, and resume on their next message.

The system prompt is Codex's own plus AUG's session prompt (developer instructions),
built once per session and kept in its record, like on LangGraph.  AUG's tools reach
Codex as dynamic tools, answered in this process; Codex's own tools run without
approval, behind the guard hook.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from langchain_core.tools import BaseTool

from aug.core.harness.base import Harness
from aug.core.harness.codex.runtime import CodexRuntime, CodexUnusable, codex_runtime
from aug.core.harness.codex.thread import ChatThread
from aug.core.harness.codex.tools import dynamic_tools
from aug.core.harness.records import RecordStore, SessionRecord
from aug.core.harness.tool_context import ToolContext, pending_approvals
from aug.core.harness.types import (
    Address,
    ApprovalDecision,
    ContextUsage,
    FilePart,
    HarnessError,
    HarnessEvent,
    Message,
    ModelTier,
    SessionStatus,
    TextPart,
    TurnContext,
    TurnEnded,
    UnsolicitedSink,
)
from aug.core.prompts import (
    HARNESS_COMPACT_STOPPED,
    HARNESS_COMPACT_TIMED_OUT,
    USER_FILE_HINT,
    USER_IMAGE_HINT,
)
from aug.core.reflexes import Reflex
from aug.core.workspace import session_prompt
from aug.utils.codex_app_server import AppServerClosed, CodexRpcError, JsonObject
from aug.utils.data import WORKSPACE_DIR
from aug.utils.media import MODEL_IMAGE_TYPES

logger = logging.getLogger(__name__)

# How long a compaction may take before it is stopped.
COMPACT_SECONDS = 10 * 60
# Idle threads kept open on the app-server at most; busy ones never count.
MAX_OPEN_THREADS = 16
# How long asking for the plan's usage limits may hold up /status.
_LIMITS_SECONDS = 5
# How long a oneshot() may take.
_ONESHOT_SECONDS = 15 * 60
# What Codex says when asked to resume a thread it has no file of.
_THREAD_GONE = "no rollout found"
# Codex errors that mean it is not there to run a turn.
_CODEX_DOWN = (CodexRpcError, AppServerClosed, TimeoutError, OSError)
# Per-thread settings for oneshot(): a bare model call, no tools, nothing but the prompt.
_ONESHOT_CONFIG = {
    "web_search": "disabled",
    "features.shell_tool": False,
    "features.multi_agent": False,
    "features.multi_agent_v2": False,
    "features.view_image": False,
    "include_environment_context": False,
    "include_permissions_instructions": False,
    "include_collaboration_mode_instructions": False,
    "skills.include_instructions": False,
}


@dataclass(frozen=True)
class CodexProfile:
    """One Codex agent version.

    model / effort: Codex's model and reasoning effort for the profile.
    tools:          AUG's tools, the MCP manager's included.
    always_loaded:  names of *tools* in every request; the rest wait in the ``aug``
                    namespace until Codex looks them up.
    models:         oneshot()'s model per tier.
    reflexes:       run on each message before Codex sees it (see Harness.run).
    """

    model: str
    effort: str
    tools: Sequence[BaseTool]
    always_loaded: frozenset[str] = frozenset()
    models: Mapping[ModelTier, str] = field(default_factory=dict)
    reflexes: Sequence[Reflex] = ()

    def __post_init__(self) -> None:
        if unknown := self.always_loaded - {t.name for t in self.tools}:
            raise ValueError(f"always_loaded names tools the profile lacks: {sorted(unknown)}")


@dataclass(frozen=True, kw_only=True)
class _Record(SessionRecord):
    """A chat's Codex thread: its id, empty until Codex has started it."""

    thread_id: str = ""


class CodexHarness(Harness):
    def __init__(self, profile: CodexProfile, *, runtime: CodexRuntime = codex_runtime) -> None:
        self.profile = profile
        self._runtime = runtime
        self._records = RecordStore(runtime.home / "aug-threads", _Record)
        self._threads: dict[str, ChatThread] = {}
        self._opening: dict[str, asyncio.Lock] = {}
        # Injected before the turn could take it: steered in once it can.
        self._early: dict[str, list[tuple[Message, list[JsonObject]]]] = {}

    # -- Harness -----------------------------------------------------------------------

    @property
    def reflexes(self) -> Sequence[Reflex]:
        return self.profile.reflexes

    async def _run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        run = self._active_run(context_id)
        stopped = run.user_requested_stop if run is not None else asyncio.Event()
        try:
            items = await _render(message)
            signed_in = await self._signed_in()
            thread = await self._thread(context_id, turn.reply_to) if signed_in else None
        except Exception as e:
            yield TurnEnded(
                error=_opening_error(context_id, e), unconsumed=self._take_early(context_id)
            )
            return
        if thread is None:
            yield TurnEnded(
                error=HarnessError(kind="login", login_with="codex"),
                unconsumed=self._take_early(context_id),
            )
            return
        if stopped.is_set():
            yield TurnEnded(cancelled=True, unconsumed=self._take_early(context_id))
            return
        ctx = ToolContext(context_id=context_id, turn=turn, stopped=stopped, _oneshot=self.oneshot)
        async with aclosing(await thread.start(items, ctx, self._turn_params())) as events:
            # Steered in order (one injected meanwhile joins the queue); one the turn
            # refuses comes back as unconsumed.
            refused: list[Message] = []
            early = self._early.get(context_id, [])
            while early:
                injected, injected_items = early.pop(0)
                if not await thread.steer(injected_items):
                    refused.append(injected)
            self._early.pop(context_id, None)  # from here on inject() steers directly
            async for event in events:
                if isinstance(event, TurnEnded) and refused:
                    event = replace(event, unconsumed=[*event.unconsumed, *refused])
                yield event

    async def inject(self, message: Message, context_id: str) -> bool:
        run = self._active_run(context_id)
        if run is None or not run.accepting:
            return False
        items = await _render(message)
        early = self._early.get(context_id)
        if early is not None:  # the turn is not taking messages yet: in order, later
            early.append((message, items))
            return True
        thread = self._threads.get(context_id)
        return thread is not None and await thread.steer(items)

    async def cancel(self, context_id: str) -> bool:
        run = self._active_run(context_id)
        thread = self._threads.get(context_id)
        if run is None:
            # A turn no run owns (one Codex started, a compaction) stops all the same.
            if thread is not None and thread.turn_open:
                logger.info("cancel context=%s (no run)", context_id)
                return await thread.interrupt()
            return False
        logger.info("cancel context=%s run=%s", context_id, run.id)
        run.request_stop()
        pending_approvals.deny_all(context_id, ApprovalDecision.DENIED)
        if thread is not None:
            await thread.interrupt()
        return True

    async def compact(self, context_id: str) -> bool:
        record = await self._records.load(context_id)
        if record is None or not record.thread_id:
            return False
        thread = await self._thread(context_id, record.reply_to)
        ctx = ToolContext(
            context_id=context_id,
            turn=TurnContext(reply_to=record.reply_to, attended=False),
            _oneshot=self.oneshot,
        )
        ended: TurnEnded | None = None
        try:
            async with asyncio.timeout(COMPACT_SECONDS):
                async for event in await thread.compact(ctx):
                    if isinstance(event, TurnEnded):
                        ended = event
        except TimeoutError:
            logger.warning("codex compaction timed out context=%s", context_id)
            await thread.interrupt()
            raise ValueError(HARNESS_COMPACT_TIMED_OUT) from None
        if ended is not None and ended.cancelled:
            raise ValueError(HARNESS_COMPACT_STOPPED)
        ok = ended is not None and ended.error is None
        if ok:
            await self._remember(context_id, None)  # what was stored is from before
        return ok

    async def session_prompt(self, context_id: str, interface: str) -> str:
        record = await self._records.load(context_id)
        if record is not None:
            return record.prompt
        return await asyncio.to_thread(session_prompt, interface, skills_index=False)

    async def session_status(self, context_id: str) -> SessionStatus:
        """The context from the last model call (once the thread is closed, from what it
        last reported); the limits are the plan's, the latest the runtime heard of."""
        record = await self._records.load(context_id)
        thread = self._threads.get(context_id)
        context: ContextUsage | None = None
        if record is not None and record.thread_id:
            context = thread.context if thread is not None else None
            if context is not None:
                await self._remember(context_id, context)
            else:
                context = record.context
        if not self._runtime.limits:
            try:
                async with asyncio.timeout(_LIMITS_SECONDS):
                    await self._runtime.refresh_limits()
            except _CODEX_DOWN as e:
                logger.info("codex usage limits unavailable: %r", e)
        return SessionStatus(
            model=self.profile.model,
            context=context,
            limits=self._runtime.limits,
            background_tasks=thread.background_tasks if thread is not None else 0,
        )

    async def oneshot(
        self, prompt: str, tier: ModelTier = ModelTier.STRONG, *, system: str = ""
    ) -> str:
        folder = self._runtime.home / "oneshot"
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
        params: JsonObject = {
            "model": self.profile.models.get(tier) or self.profile.model,
            "cwd": str(folder),
            "ephemeral": True,
            # Nothing it is told can make it write anywhere.
            "sandbox": "read-only",
            "config": _ONESHOT_CONFIG,
        }
        if system:
            params["baseInstructions"] = system
        started = await self._runtime.request("thread/start", params)
        listener = _Oneshot()
        thread_id = str(started["thread"]["id"])
        self._runtime.listen(thread_id, listener)
        turn_id = ""
        try:
            result = await self._runtime.request(
                "turn/start",
                {"threadId": thread_id, "input": [{"type": "text", "text": prompt}]},
            )
            turn_id = str((result.get("turn") or {}).get("id") or "")
            async with asyncio.timeout(_ONESHOT_SECONDS):
                return await listener.result
        except TimeoutError:
            if turn_id:  # it would go on running, and spending, for nobody
                with contextlib.suppress(*_CODEX_DOWN):
                    await self._runtime.request(
                        "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
                    )
            raise
        finally:
            self._runtime.unload(thread_id, listener)  # jobs call oneshot() every night

    @asynccontextmanager
    async def running(self, on_unsolicited: UnsolicitedSink) -> AsyncIterator[None]:
        self._sink = on_unsolicited
        try:
            async with self._runtime.using():
                yield
        finally:
            for thread in self._threads.values():
                thread.close()
            self._threads.clear()
            self._sink = None

    async def _context_after_turn(self, context_id: str) -> ContextUsage | None:
        thread = self._threads.get(context_id)
        context = thread.context if thread is not None else None
        if context is not None:
            await self._remember(context_id, context)
        return context

    def _open_turn(self, context_id: str, turn: TurnContext) -> None:
        self._early[context_id] = []  # inject() works from TurnStarted on

    def _close_turn(self, context_id: str) -> None:
        self._early.pop(context_id, None)

    # -- threads -----------------------------------------------------------------------

    async def _thread(self, context_id: str, reply_to: Address) -> ChatThread:
        """The chat's Codex thread, open on the running app-server: started, or
        resumed if the app-server does not have it open.  Nothing is awaited between
        handing it out and its caller asking for a turn, so it is never let go of in
        between."""
        self._let_go_of_idle(keep=context_id)
        lock = self._opening.setdefault(context_id, asyncio.Lock())
        async with lock:
            record = await self._records.load(context_id) or await self._new_record(
                context_id, reply_to
            )
            thread = self._threads.get(context_id)
            if thread is None:
                thread = ChatThread(
                    context_id,
                    self._runtime,
                    tools=self._tools_by_name,
                    reply_to=reply_to,
                    on_unsolicited=lambda events: self._deliver_unsolicited(context_id, events),
                    oneshot=self.oneshot,
                )
                self._threads[context_id] = thread
            thread.reply_to = reply_to
            if record.thread_id and not self._runtime.is_loaded(record.thread_id):
                record = await self._resume(record)
            if not record.thread_id:
                record = await self._start(record)
            thread.bind(record.thread_id)
        return thread

    async def _start(self, record: _Record) -> _Record:
        result = await self._runtime.request(
            "thread/start",
            {
                **self._thread_params(record),
                "dynamicTools": dynamic_tools(self.profile.tools, self.profile.always_loaded),
            },
        )
        record = replace(record, thread_id=str(result["thread"]["id"]), context=None)
        await self._records.save(record)
        self._runtime.mark_loaded(record.thread_id)
        logger.info("codex thread started context=%s", record.context_id)
        return record

    async def _resume(self, record: _Record) -> _Record:
        """Open the thread on the app-server again.  One Codex no longer has (a wiped
        CODEX_HOME) comes back without its id, to be started afresh."""
        await self._runtime.unloaded(record.thread_id)
        try:
            await self._runtime.request(
                "thread/resume", {"threadId": record.thread_id, **self._thread_params(record)}
            )
        except CodexRpcError as e:
            if _THREAD_GONE not in e.message:
                raise
            logger.warning(
                "codex thread gone, starting afresh context=%s: %s", record.context_id, e
            )
            return replace(record, thread_id="")
        self._runtime.mark_loaded(record.thread_id)
        logger.info("codex thread resumed context=%s", record.context_id)
        return record

    def _let_go_of_idle(self, *, keep: str) -> None:
        """Let go of the least recently used idle threads past MAX_OPEN_THREADS: the
        app-server unloads them, and they resume on their next message.  One being
        opened or running a turn is never let go of."""
        idle = sorted(
            (t for cid, t in self._threads.items() if cid != keep and self._idle(t)),
            key=lambda t: t.last_used,
        )
        for thread in idle[: max(0, len(idle) - MAX_OPEN_THREADS)]:
            del self._threads[thread.context_id]
            thread.close()
            if thread.thread_id is not None:
                self._runtime.unload(thread.thread_id, thread)

    def _idle(self, thread: ChatThread) -> bool:
        lock = self._opening.get(thread.context_id)
        return not thread.busy and not (lock is not None and lock.locked())

    def _thread_params(self, record: _Record) -> JsonObject:
        return {
            "model": self.profile.model,
            "cwd": str(WORKSPACE_DIR),
            "developerInstructions": record.prompt,
        }

    def _turn_params(self) -> JsonObject:
        return {"model": self.profile.model, "effort": self.profile.effort}

    def _take_early(self, context_id: str) -> list[Message]:
        """The messages injected for a turn that never started, to hand back."""
        return [message for message, _ in self._early.pop(context_id, None) or []]

    def _tools_by_name(self) -> dict[str, BaseTool]:
        return {t.name: t for t in self.profile.tools}

    async def _signed_in(self) -> bool:
        result = await self._runtime.request("account/read", {"refreshToken": False})
        return result.get("account") is not None or not result.get("requiresOpenaiAuth")

    def _reply_to(self, context_id: str) -> Address | None:
        thread = self._threads.get(context_id)
        return thread.reply_to if thread is not None else None

    # -- records -----------------------------------------------------------------------

    async def _new_record(self, context_id: str, reply_to: Address) -> _Record:
        record = _Record(
            context_id=context_id,
            interface=reply_to.interface,
            prompt=await asyncio.to_thread(session_prompt, reply_to.interface, skills_index=False),
            conversation_id=reply_to.conversation_id,
            chat_id=reply_to.chat_id,
        )
        await self._records.save(record)
        return record

    async def _remember(self, context_id: str, context: ContextUsage | None) -> None:
        """Keep *context* for when the thread is closed; None forgets a stale one."""
        await self._records.update(context_id, lambda r: replace(r, context=context))


class _Oneshot:
    """Listens to a oneshot()'s thread: its answer is the last thing it says."""

    context_id = ""  # no chat's

    def __init__(self) -> None:
        self.result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._text = ""

    def on_notification(self, method: str, params: JsonObject) -> None:
        item = params.get("item") or {}
        if method == "item/completed" and item.get("type") == "agentMessage":
            self._text = str(item.get("text") or "")
        elif method == "turn/completed" and not self.result.done():
            turn = params.get("turn") or {}
            if turn.get("status") == "completed":
                self.result.set_result(self._text)
            else:
                error = (turn.get("error") or {}).get("message")
                self.result.set_exception(RuntimeError(f"oneshot failed: {error}"))

    async def on_tool_call(self, params: JsonObject) -> JsonObject:
        raise CodexRpcError(-32603, "oneshot has no tools")

    def on_closed(self) -> None:
        if not self.result.done():
            self.result.set_exception(RuntimeError("oneshot: the Codex process exited"))


def _opening_error(context_id: str, error: Exception) -> HarnessError:
    """Why a turn could not start: Codex is down or won't run AUG's guard
    (``unavailable``), or something unexpected broke (``internal``)."""
    if isinstance(error, (*_CODEX_DOWN, CodexUnusable)):
        logger.warning("codex unavailable context=%s: %r", context_id, error)
        return HarnessError(kind="unavailable")
    logger.exception("codex thread could not open context=%s", context_id)
    return HarnessError(kind="internal")


async def _render(message: Message) -> list[JsonObject]:
    """*message* as Codex user input: the time first, images by path with a hint
    naming it, other files as a hint with their path."""
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    items: list[JsonObject] = [{"type": "text", "text": f"[{now}]"}]
    for part in message.parts:
        match part:
            case TextPart(text=text) if text:
                items.append({"type": "text", "text": text})
            case FilePart(path=path, media_type=media_type) if media_type in MODEL_IMAGE_TYPES:
                if await asyncio.to_thread(path.is_file):
                    items.append({"type": "localImage", "path": str(path)})
                else:
                    logger.warning("image unreadable: %s", path)
                items.append({"type": "text", "text": USER_IMAGE_HINT.format(path=path)})
            case FilePart(path=path, media_type=media_type, filename=filename):
                hint = USER_FILE_HINT.format(filename=filename, media_type=media_type, path=path)
                items.append({"type": "text", "text": hint})
    return items
