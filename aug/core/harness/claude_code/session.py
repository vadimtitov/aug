"""One chat's warm Claude Code process, and the one reader of everything it says.

The reader turns the SDK's messages into harness events and hands them to the turn in
flight.  A turn ends at its result followed by the CLI going idle, not at the first
result: a message injected late, or a tool result still landing, belongs to it too.
When the CLI starts a turn by itself (background work finishing after the turn ended),
the reader opens an unattended turn and passes its events to ``on_unsolicited``.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ContextUsageResponse,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import TERMINAL_TASK_STATUSES

from aug.core.harness.claude_code.tools import ToolCalls, canonical_name
from aug.core.harness.tool_context import ToolContext
from aug.core.harness.types import (
    Address,
    AgentEvent,
    ContextUsage,
    HarnessError,
    HarnessEvent,
    LimitUsage,
    StatusUpdate,
    TextDelta,
    ToolEnded,
    ToolStarted,
    TurnContext,
    TurnEnded,
    Usage,
)
from aug.core.prompts import CLAUDE_COMPACTED_STATUS

logger = logging.getLogger(__name__)

# How long an interrupt may take before the turn is ended without the CLI.
_INTERRUPT_SECONDS = 10
# How long the CLI may take to report its context before the question is dropped.
_CONTEXT_SECONDS = 5
# Context categories that hold no tokens of the conversation's own.
_NOT_IN_CONTEXT = frozenset({"Free space", "Autocompact buffer"})

# A user message's content: Anthropic content blocks.
Content = list[dict[str, Any]]
# Receives the events of a turn the CLI started by itself.
UnsolicitedHandler = Callable[[AsyncIterator[HarnessEvent]], Awaitable[None]]


@dataclass
class _Turn:
    unsolicited: bool
    events: asyncio.Queue[HarnessEvent | None] = field(default_factory=asyncio.Queue)
    ran: bool = False  # the CLI reported "running" during this turn
    result_seen: bool = False
    error: HarnessError | None = None
    cancelled: bool = False
    usage: Usage = field(default_factory=Usage)


class ChatSession:
    """One chat's Claude Code process.  Opened by the harness, which also closes it."""

    def __init__(
        self,
        context_id: str,
        options: ClaudeAgentOptions,
        calls: ToolCalls,
        *,
        fingerprint: str,
        reply_to: Address,
        on_unsolicited: UnsolicitedHandler,
        limits: dict[str, LimitUsage] | None = None,
    ) -> None:
        """*limits* is updated in place whenever the CLI reports the plan's usage
        windows; the harness shares one across its sessions, as the plan is one."""
        self.context_id = context_id
        self.fingerprint = fingerprint
        self.reply_to = reply_to
        self.last_used = time.monotonic()
        self.session_id: str | None = None  # the CLI's, once it says
        self._options = options
        self._calls = calls
        self._on_unsolicited = on_unsolicited
        self._limits = limits if limits is not None else {}
        self._client: ClaudeSDKClient | None = None
        self._reader: asyncio.Task[None] | None = None
        self._turn: _Turn | None = None
        self._state: str | None = None  # the CLI's session state: running / idle / …
        self._tasks: set[str] = set()  # tasks started and not yet finished
        self._tool_names: dict[str, str] = {}  # tool_use_id → canonical name
        self._streamed: set[str] = set()  # assistant message ids whose text was streamed
        self._message_id: dict[str | None, str] = {}  # parent → message being streamed
        self._deliveries: set[asyncio.Task[None]] = set()  # self-started turns being sent
        # The context as of the last main-thread model call: its tokens in and out are
        # the next call's context.  None once a compaction made it stale.  The output
        # count is final only at the call's message_delta.
        self._context_tokens: int | None = None
        self._output_tokens = 0
        self._model = ""  # the model that made that call, as the API names it
        self._window: int | None = None  # its context window, from the turn's result
        self._window_warned = False
        self._compacts_at: int | None = None  # from the CLI's last context report
        # The CLI's own report is wanted: none yet for this process, or a compaction
        # since.  It carries what a model call's usage doesn't: the breakdown and where
        # the CLI compacts.
        self.needs_report = True

    @property
    def alive(self) -> bool:
        return self._reader is not None and not self._reader.done()

    @property
    def busy(self) -> bool:
        """A turn is in flight, or background work is still running: never evict."""
        return self._turn is not None or self._state == "running" or bool(self._tasks)

    @property
    def accepting(self) -> bool:
        """An attended turn is in flight and can take an injected message."""
        return self._turn is not None and not self._turn.unsolicited and not self._turn.cancelled

    async def open(self) -> None:
        client = ClaudeSDKClient(self._options)
        await client.connect()
        self._client = client
        self._reader = asyncio.create_task(self._read(), name=f"claude-{self.context_id}")
        logger.info("claude process opened context=%s", self.context_id)

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                logger.warning(
                    "claude disconnect failed context=%s", self.context_id, exc_info=True
                )
        self._client = None
        logger.info("claude process closed context=%s", self.context_id)

    async def start(
        self, content: Content, ctx: ToolContext, *, verbatim: bool = True
    ) -> AsyncIterator[HarnessEvent]:
        """Send *content*; returns the turn it starts, as events ending with TurnEnded.

        *ctx* is the ToolContext AUG's tools get; their events go into this stream.
        *verbatim* False lets the CLI read the text as a slash command (``/compact``).
        Once this returns, inject() reaches the turn.
        """
        assert self._client is not None, "open() first"
        if self._turn is not None:
            # A turn the CLI started by itself just before this one: this turn takes it
            # over, so its events reach the user who is here now.
            self._end(self._turn)
        turn = _Turn(unsolicited=False, ran=self._state == "running")
        self._turn = turn
        self._calls.context = replace(ctx, _emit=self._emit)
        self.last_used = time.monotonic()
        await self._client.query(_as_messages(content, verbatim=verbatim))
        return _drain(turn)

    async def inject(self, content: Content) -> bool:
        """Add *content* to the attended turn in flight; False if there is none."""
        if not self.accepting or self._client is None:
            return False
        await self._client.query(_as_messages(content, verbatim=True))
        return True

    @property
    def background_tasks(self) -> int:
        """Background tasks started and not yet finished."""
        return len(self._tasks)

    @property
    def context(self) -> ContextUsage | None:
        """The context as the last model call left it: free, no breakdown.  None
        before the first call, or after a compaction until the next one."""
        if self._context_tokens is None or not self._window:
            return None
        return ContextUsage(
            used=self._context_tokens + self._output_tokens,
            window=self._window,
            compacts_at=self._compacts_at,
        )

    async def context_usage(self) -> ContextUsage | None:
        """How full the CLI says the context is, with its breakdown; None when the
        process is gone or doesn't answer in time.  The CLI counts it with the API's
        token counting, a request per category: for when someone asks, not per turn."""
        if self._client is None or not self.alive:
            return None
        self.needs_report = False  # asked: a failed report isn't retried every turn
        try:
            raw = await asyncio.wait_for(self._client.get_context_usage(), _CONTEXT_SECONDS)
            context = _context_usage(raw)
        except Exception as e:
            logger.warning("claude context usage failed context=%s: %r", self.context_id, e)
            return None
        self._compacts_at = context.compacts_at
        self._window = self._window or context.window
        return context

    @property
    def has_background_work(self) -> bool:
        """Background tasks still running, or a turn the CLI started by itself."""
        return bool(self._tasks) or self.running_unsolicited

    @property
    def running_unsolicited(self) -> bool:
        """A turn the CLI started by itself is in flight."""
        return self._turn is not None and self._turn.unsolicited

    async def interrupt(self) -> bool:
        """Stop the turn in flight.  It ends with TurnEnded(cancelled=True), here if
        the CLI does not answer the interrupt."""
        turn = self._turn
        if turn is None or self._client is None:
            return False
        turn.cancelled = True
        if self._calls.context is not None:
            self._calls.context.stopped.set()
        try:
            await asyncio.wait_for(self._client.interrupt(), _INTERRUPT_SECONDS)
        except Exception as e:
            logger.warning("claude interrupt failed context=%s: %s", self.context_id, e)
            self._end(turn)
        return True

    async def _emit(self, event: AgentEvent) -> None:
        if self._turn is not None:
            self._turn.events.put_nowait(event)

    async def _read(self) -> None:
        assert self._client is not None
        failed = False
        try:
            async for message in self._client.receive_messages():
                self.last_used = time.monotonic()
                self._handle(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(
                "claude process failed context=%s: %s: %s", self.context_id, type(e).__name__, e
            )
            failed = True
        finally:
            if self._turn is not None:
                turn = self._turn
                if failed or not turn.result_seen:
                    turn.error = turn.error or HarnessError(kind="unavailable")
                self._end(turn)
            self._tasks.clear()
            logger.info("claude process ended context=%s", self.context_id)

    def _handle(self, message: Any) -> None:
        match message:
            case StreamEvent():
                self._on_stream_event(message)
            case AssistantMessage():
                self._on_assistant(message)
            case UserMessage():
                self._on_user(message)
            case ResultMessage():
                self._on_result(message)
            case RateLimitEvent():
                self._on_rate_limit(message)
            case TaskStartedMessage(task_id=task_id):
                self._tasks.add(task_id)
            case TaskNotificationMessage(task_id=task_id):
                self._tasks.discard(task_id)
            case TaskUpdatedMessage(task_id=task_id, patch=patch):
                if patch.get("status") in TERMINAL_TASK_STATUSES:
                    self._tasks.discard(task_id)
            case SystemMessage(subtype="session_state_changed", data=data):
                self._on_state(data.get("state"))
            case SystemMessage(subtype="init", data=data):
                self.session_id = data.get("session_id") or self.session_id
            case SystemMessage(subtype="compact_boundary"):
                self._context_tokens = None
                self.needs_report = True
                self._put(StatusUpdate(text=CLAUDE_COMPACTED_STATUS))

    def _on_state(self, state: str | None) -> None:
        """A turn ends when the CLI goes idle after running for it, result or not."""
        self._state = state
        turn = self._turn
        if state == "running":
            if turn is None:
                self._start_unsolicited()
            else:
                turn.ran = True
        elif state == "idle" and turn is not None and turn.ran:
            self._end(turn)

    def _start_unsolicited(self) -> None:
        logger.info("claude started a turn by itself context=%s", self.context_id)
        turn = _Turn(unsolicited=True, ran=True)
        self._turn = turn
        self._calls.context = ToolContext(
            context_id=self.context_id,
            turn=TurnContext(reply_to=self.reply_to, attended=False),
            _emit=self._emit,
        )

        task = asyncio.create_task(
            self._on_unsolicited(_drain(turn)), name=f"claude-push-{self.context_id}"
        )
        self._deliveries.add(task)
        task.add_done_callback(self._deliveries.discard)

    def _end(self, turn: _Turn) -> None:
        # The context is the harness's to add (_context_after_turn): it decides
        # whether the CLI's own report is worth asking for.
        turn.events.put_nowait(
            TurnEnded(error=turn.error, cancelled=turn.cancelled, usage=turn.usage)
        )
        turn.events.put_nowait(None)
        if self._turn is turn:
            self._turn = None
            self._calls.clear()

    def _put(self, event: HarnessEvent) -> None:
        if self._turn is not None:
            self._turn.events.put_nowait(event)

    def _on_stream_event(self, message: StreamEvent) -> None:
        event = message.event
        kind = event.get("type")
        if kind == "message_start":
            if message.parent_tool_use_id is None:
                self._output_tokens = 0  # a new main-thread call
            self._message_id[message.parent_tool_use_id] = (event.get("message") or {}).get(
                "id", ""
            )
        elif kind == "message_delta" and message.parent_tool_use_id is None:
            output = (event.get("usage") or {}).get("output_tokens")
            if output is not None:
                self._output_tokens = max(self._output_tokens, int(output))
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                self._streamed.add(self._message_id.get(message.parent_tool_use_id, ""))
                self._put(TextDelta(text=delta["text"], parent_id=message.parent_tool_use_id))

    def _on_assistant(self, message: AssistantMessage) -> None:
        if message.error is not None and self._turn is not None:
            self._turn.error = _assistant_error(message)
        tokens = _input_tokens(message.usage or {})
        # A message the CLI made up itself (an API error's text, say) counts nothing.
        if message.parent_tool_use_id is None and tokens and message.error is None:
            self._context_tokens = tokens
            # Whichever of this and the call's message_delta comes last, keep the larger.
            output = int((message.usage or {}).get("output_tokens") or 0)
            self._output_tokens = max(self._output_tokens, output)
            self._model = message.model
        streamed = message.message_id in self._streamed
        for block in message.content:
            match block:
                case TextBlock(text=text) if not streamed and text:
                    # Not streamed: the CLI's own messages (an API error, say).
                    self._put(TextDelta(text=text, parent_id=message.parent_tool_use_id))
                case ToolUseBlock(id=tool_id, name=name, input=args):
                    self._calls.announce(tool_id, name, args)
                    self._tool_names[tool_id] = canonical_name(name)
                    self._put(
                        ToolStarted(
                            id=tool_id,
                            name=canonical_name(name),
                            args=args,
                            parent_id=message.parent_tool_use_id,
                        )
                    )

    def _on_user(self, message: UserMessage) -> None:
        if not isinstance(message.content, list):
            return
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                self._put(
                    ToolEnded(
                        id=block.tool_use_id,
                        name=self._tool_names.pop(block.tool_use_id, ""),
                        output=_result_text(block.content),
                        error=bool(block.is_error),
                        parent_id=message.parent_tool_use_id,
                    )
                )

    def _on_result(self, message: ResultMessage) -> None:
        self.session_id = message.session_id or self.session_id
        self._window = _context_window(message.model_usage or {}, self._model) or self._window
        if self._window is None and self._model and message.model_usage and not self._window_warned:
            self._window_warned = True
            logger.warning(
                "claude context window unknown context=%s model=%s usage keys=%s",
                self.context_id,
                self._model,
                sorted(message.model_usage),
            )
        turn = self._turn
        if turn is None:
            return
        turn.result_seen = True
        turn.usage = _add_usage(turn.usage, message.usage or {})
        if message.terminal_reason in ("aborted_streaming", "aborted_tools"):
            turn.cancelled = True
        elif message.is_error and not turn.cancelled:
            turn.error = turn.error or _result_error(message)
        if self._state != "running":
            # Answered without running (a local slash command), or a CLI that reports
            # no session state: no idle is coming.
            self._end(turn)

    def _on_rate_limit(self, message: RateLimitEvent) -> None:
        info = message.rate_limit_info
        self._limits.update({limit.window: limit for limit in _limits(info)})
        turn = self._turn
        if turn is None:
            return
        if info.status == "rejected":
            retry_at = datetime.fromtimestamp(info.resets_at, UTC) if info.resets_at else None
            turn.error = HarnessError(kind="rate_limited", retry_at=retry_at)


async def _drain(turn: _Turn) -> AsyncIterator[HarnessEvent]:
    while (event := await turn.events.get()) is not None:
        yield event


def _as_messages(content: Content, *, verbatim: bool) -> AsyncIterator[dict[str, Any]]:
    async def messages() -> AsyncIterator[dict[str, Any]]:
        message: dict[str, Any] = {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
        }
        if verbatim:
            # Delivered as written: no @path expansion, no slash commands.
            message["client_composed"] = True
        yield message

    return messages()


def _result_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return str(content)


def _add_usage(usage: Usage, raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens + int(raw.get("input_tokens") or 0),
        output_tokens=usage.output_tokens + int(raw.get("output_tokens") or 0),
        cache_read_tokens=usage.cache_read_tokens + int(raw.get("cache_read_input_tokens") or 0),
        cache_write_tokens=usage.cache_write_tokens
        + int(raw.get("cache_creation_input_tokens") or 0),
    )


def _input_tokens(usage: dict[str, Any]) -> int:
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(int(usage.get(key) or 0) for key in keys)


def _context_window(model_usage: dict[str, Any], model: str) -> int | None:
    """*model*'s context window from a result's per-model usage.  Its keys are the
    CLI's model names (``claude-haiku-4-5``), which *model*, the API's
    (``claude-haiku-4-5-20251001``), extends."""
    names = [n for n in model_usage if model and (model == n or model.startswith(f"{n}-"))]
    if not names:
        return None
    usage = model_usage[max(names, key=len)]  # the exact name, else the longest prefix
    window = usage.get("contextWindow") if isinstance(usage, dict) else None
    return int(window) if window else None


def _context_usage(raw: ContextUsageResponse) -> ContextUsage:
    window = int(raw.get("rawMaxTokens") or raw.get("maxTokens") or 0)
    limit = int(raw.get("maxTokens") or 0)
    auto = bool(raw.get("isAutoCompactEnabled"))
    breakdown = [
        (c["name"].lower(), int(c["tokens"]))
        for c in raw.get("categories") or []
        if c["name"] not in _NOT_IN_CONTEXT and not c.get("isDeferred") and c.get("tokens")
    ]
    return ContextUsage(
        used=int(raw.get("totalTokens") or 0),
        window=window,
        # maxTokens is the window less the autocompact buffer, when the CLI reserves one.
        compacts_at=limit if auto and 0 < limit < window else None,
        breakdown=tuple(sorted(breakdown, key=lambda item: -item[1])),
    )


def _limits(info: RateLimitInfo) -> list[LimitUsage]:
    """Every usage window the event reports: all of them when the CLI sends its
    ``unifiedWindows``, else the one the event is about."""
    windows = info.raw.get("unifiedWindows")
    if isinstance(windows, dict) and windows:
        return [
            LimitUsage(
                window=name,
                utilization=float(w["utilization"]),
                resets_at=_timestamp(w.get("resetsAt")),
            )
            for name, w in windows.items()
            if isinstance(w, dict) and w.get("utilization") is not None
        ]
    if info.rate_limit_type is None or info.utilization is None:
        return []
    return [
        LimitUsage(
            window=info.rate_limit_type,
            utilization=info.utilization,
            resets_at=_timestamp(info.resets_at),
        )
    ]


def _timestamp(value: Any) -> datetime | None:
    return datetime.fromtimestamp(value, UTC) if isinstance(value, int | float) else None


def _assistant_error(message: AssistantMessage) -> HarnessError:
    text = " ".join(b.text for b in message.content if isinstance(b, TextBlock)).lower()
    logger.warning("claude assistant error %s: %.200s", message.error, text)
    match message.error:
        case "authentication_failed":
            return HarnessError(kind="login")
        case "billing_error":
            return HarnessError(kind="auth")
        case "rate_limit":
            return HarnessError(kind="rate_limited")
        case "server_error":
            return HarnessError(kind="unavailable")
        case "invalid_request" if "too long" in text or "context" in text:
            return HarnessError(kind="context_full")
    return HarnessError(kind="internal")


def _result_error(message: ResultMessage) -> HarnessError:
    logger.warning(
        "claude result error subtype=%s status=%s errors=%s",
        message.subtype,
        message.api_error_status,
        message.errors,
    )
    status = message.api_error_status
    if message.subtype == "error_max_turns":
        return HarnessError(kind="step_limit")
    if status == 429:
        return HarnessError(kind="rate_limited")
    if status in (401, 403):
        return HarnessError(kind="login")
    if status is not None and status >= 500:
        return HarnessError(kind="unavailable")
    return HarnessError(kind="internal")
