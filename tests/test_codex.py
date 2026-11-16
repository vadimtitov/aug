"""The Codex harness's own logic, with Codex's notifications scripted.

The real app-server is exercised in test_codex_cli.py; here are what a fake model can't
easily make it do: subagents, usage limits, a turn Codex starts by itself, a dying or
stuck process, plus the guard and the runtime's bookkeeping.  The login is in
test_codex_login.py.
"""

import asyncio
import io
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool

from aug.core.harness import (
    Address,
    HarnessError,
    Message,
    Sender,
    StatusUpdate,
    TextDelta,
    TextPart,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnContext,
    TurnEnded,
)
from aug.core.harness.base import error_text
from aug.core.harness.codex import guard
from aug.core.harness.codex import harness as codex_harness
from aug.core.harness.codex import thread as codex_thread
from aug.core.harness.codex.harness import CodexHarness, CodexProfile
from aug.core.harness.codex.runtime import CodexRuntime, CodexUnusable
from aug.core.harness.codex.thread import ChatThread, original_command
from aug.core.harness.codex.tools import dynamic_tools, run_tool, tool_answer
from aug.core.harness.tool_bridge import ToolImage, ToolOutcome
from aug.core.harness.tool_context import ToolContext
from aug.core.run import run_registry
from aug.core.tools.blocked_paths import path_block_reason
from aug.core.tools.context import current_tool_context
from aug.utils.codex_app_server import CodexRpcError
from aug.utils.data import CODEX_DIR, DATA_DIR
from aug.utils.hushed import hushed_command

ADDR = Address(interface="telegram", conversation_id="tg-1", chat_id="1")
THREAD = "thread-1"
CHILD = "thread-child"


class FakeRuntime:
    """Answers requests from a script and records them."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path("/nonexistent")
        self.requests: list[tuple[str, dict]] = []
        self.answers: dict[str, Any] = {
            "turn/start": {"turn": {"id": "turn-1"}},
            "thread/start": {"thread": {"id": THREAD}},
            "account/read": {"account": {"type": "chatgpt"}},
        }
        self.listeners: dict[str, Any] = {}
        self.loaded: set[str] = set()
        self.reset = datetime.now(UTC) + timedelta(hours=2)
        self.limits: tuple = ()

    async def request(self, method: str, params: dict | None = None) -> dict:
        self.requests.append((method, params or {}))
        answer = self.answers.get(method, {})
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def listen(self, thread_id: str, listener: Any) -> bool:
        current = self.listeners.get(thread_id)
        if current not in (None, listener) and current.context_id != listener.context_id:
            return False
        self.listeners[thread_id] = listener
        return True

    def forget(self, thread_id: str, listener: Any) -> None:
        if self.listeners.get(thread_id) is listener:
            del self.listeners[thread_id]

    def is_loaded(self, thread_id: str) -> bool:
        return thread_id in self.loaded

    async def unloaded(self, thread_id: str) -> None:
        pass

    def mark_loaded(self, thread_id: str) -> None:
        self.loaded.add(thread_id)

    def unload(self, thread_id: str, listener: Any) -> None:
        self.loaded.discard(thread_id)
        self.forget(thread_id, listener)
        self.requests.append(("thread/unsubscribe", {"threadId": thread_id}))

    def retry_at(self) -> datetime | None:
        return self.reset

    def sent(self, method: str) -> list[dict]:
        return [params for m, params in self.requests if m == method]


@tool
async def progress() -> str:
    """Report progress, then finish."""
    ctx = current_tool_context()
    assert ctx is not None
    await ctx.emit(ToolProgress(text="halfway"))
    return f"done for {ctx.context_id}"


@tool
async def forever() -> str:
    """Never finish."""
    await asyncio.sleep(3600)
    return "never"


TOOLS = {t.name: t for t in (progress, forever)}


def _thread(runtime: FakeRuntime, delivered: list | None = None) -> ChatThread:
    async def on_unsolicited(events: Any) -> None:
        collected = [e async for e in events]
        if delivered is not None:
            delivered.append(collected)

    thread = ChatThread(
        "ctx", runtime, tools=lambda: TOOLS, reply_to=ADDR, on_unsolicited=on_unsolicited
    )
    thread.bind(THREAD)
    return thread


def _ctx() -> ToolContext:
    return ToolContext(context_id="ctx", turn=TurnContext(reply_to=ADDR, attended=True))


def _say(thread: ChatThread, method: str, thread_id: str = THREAD, **params: Any) -> None:
    thread.on_notification(method, {"threadId": thread_id, **params})


def _item(thread: ChatThread, phase: str, thread_id: str = THREAD, **item: Any) -> None:
    _say(thread, f"item/{phase}", thread_id, item=item)


def _done(thread: ChatThread, status: str = "completed", error: dict | None = None) -> None:
    _say(thread, "turn/completed", turn={"id": "turn-1", "status": status, "error": error})


def _call(tool_name: str, call_id: str, turn_id: str = "turn-1") -> dict:
    """Codex's item/tool/call for *tool_name* in the main thread's turn *turn_id*."""
    return {
        "threadId": THREAD,
        "turnId": turn_id,
        "tool": tool_name,
        "arguments": {},
        "callId": call_id,
    }


async def _collect(events: Any) -> list:
    return [e async for e in events]


async def _until(condition: Any) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("never happened")


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# -- turns -------------------------------------------------------------------------------


async def test_replies_stream_with_a_break_between_two_messages():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([{"type": "text", "text": "hi"}], _ctx(), {"model": "m"})
    _say(thread, "turn/started", turn={"id": "turn-1"})
    _item(thread, "started", type="agentMessage", id="a1", text="")
    _say(thread, "item/agentMessage/delta", itemId="a1", delta="Looking.")
    _item(thread, "completed", type="agentMessage", id="a1", text="Looking.")
    _item(thread, "started", type="agentMessage", id="a2", text="")
    _item(thread, "completed", type="agentMessage", id="a2", text="Done.")  # never streamed
    _done(thread)
    collected = await _collect(events)
    assert "".join(e.text for e in collected if isinstance(e, TextDelta)) == "Looking.\n\nDone."
    assert isinstance(collected[-1], TurnEnded) and collected[-1].error is None
    (params,) = runtime.sent("turn/start")
    assert params["threadId"] == THREAD and params["model"] == "m"


async def test_codex_tools_show_under_aug_names():
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    command = shlex.join(["/bin/bash", "-lc", hushed_command("ls -la /tmp")])
    _item(thread, "started", type="commandExecution", id="c1", command=command, status="inProgress")
    _item(
        thread,
        "completed",
        type="commandExecution",
        id="c1",
        command=command,
        status="completed",
        exitCode=2,
        aggregatedOutput="ls: nope",
    )
    _item(thread, "started", type="fileChange", id="f1", changes=[{"path": "/w/a.txt"}])
    _item(thread, "started", type="webSearch", id="w1", query="weather")
    _item(thread, "started", type="mcpToolCall", id="m1", server="pw", tool="click", arguments={})
    _done(thread)
    collected = await _collect(events)
    started = [(e.name, e.args) for e in collected if isinstance(e, ToolStarted)]
    assert started == [
        ("Bash", {"command": "ls -la /tmp"}),
        ("apply_patch", {"files": ["/w/a.txt"]}),
        ("web_search", {"query": "weather"}),
        ("pw__click", {}),
    ]
    (ended,) = [e for e in collected if isinstance(e, ToolEnded)]
    assert ended.error and ended.output == "ls: nope"  # a non-zero exit is an error


async def test_a_subagents_work_is_filed_under_it():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    _item(
        thread,
        "started",
        type="subAgentActivity",
        id="s1",
        kind="started",
        agentThreadId=CHILD,
        agentPath="/root/research",
    )
    assert runtime.listeners[CHILD] is thread
    assert thread.background_tasks == 1
    _say(thread, "turn/started", CHILD, turn={"id": "child-turn"})
    _item(thread, "started", CHILD, type="webSearch", id="w1", query="x")
    _say(thread, "item/agentMessage/delta", CHILD, itemId="ca", delta="found it")
    _item(thread, "completed", CHILD, type="agentMessage", id="ca", text="found it")
    _say(thread, "turn/completed", CHILD, turn={"id": "child-turn", "status": "completed"})
    assert thread.background_tasks == 0  # done, while the parent turn goes on
    _done(thread)
    collected = await _collect(events)
    (spawn,) = [e for e in collected if isinstance(e, ToolStarted) and e.name == "subagent"]
    assert spawn.id == CHILD and spawn.args == {"task": "/root/research"}
    (search,) = [e for e in collected if isinstance(e, ToolStarted) and e.name == "web_search"]
    assert search.parent_id == CHILD
    (child_text,) = [e for e in collected if isinstance(e, TextDelta)]
    assert child_text.parent_id == CHILD
    (ended,) = [e for e in collected if isinstance(e, ToolEnded)]
    assert (ended.id, ended.output) == (CHILD, "found it")


async def test_a_subagent_given_more_to_do_shows_as_a_new_call():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    spawn = {"type": "subAgentActivity", "agentThreadId": CHILD, "agentPath": "/root/r"}
    _item(thread, "started", **spawn, id="s1", kind="started")
    _say(thread, "turn/completed", CHILD, turn={"id": "c1", "status": "completed"})
    _item(thread, "started", **spawn, id="s2", kind="interacted")
    _item(thread, "started", CHILD, type="webSearch", id="w2", query="more")
    _say(thread, "turn/completed", CHILD, turn={"id": "c2", "status": "completed"})
    _done(thread)
    collected = await _collect(events)
    calls = [e.id for e in collected if isinstance(e, ToolStarted) and e.name == "subagent"]
    assert calls == [CHILD, f"{CHILD}:2"]
    (search,) = [e for e in collected if isinstance(e, ToolStarted) and e.name == "web_search"]
    assert search.parent_id == f"{CHILD}:2"
    assert runtime.listeners[CHILD] is thread  # still heard: it may be asked again


async def test_a_subagents_subagent_ends_under_its_parent():
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    _item(thread, "started", type="subAgentActivity", id="s1", kind="started", agentThreadId=CHILD)
    _item(
        thread,
        "started",
        CHILD,
        type="subAgentActivity",
        id="s2",
        kind="started",
        agentThreadId="grandchild",
    )
    _say(thread, "turn/completed", "grandchild", turn={"id": "g", "status": "completed"})
    _done(thread)
    collected = await _collect(events)
    (inner,) = [e for e in collected if isinstance(e, ToolEnded) and e.id == "grandchild"]
    assert inner.parent_id == CHILD


async def test_a_subagent_cannot_be_another_chats_thread():
    runtime = FakeRuntime()
    other = MagicMock(context_id="another chat")
    runtime.listen("thread-of-another-chat", other)
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    _item(
        thread,
        "started",
        type="subAgentActivity",
        id="s1",
        kind="started",
        agentThreadId="thread-of-another-chat",
    )
    assert runtime.listeners["thread-of-another-chat"] is other
    assert thread.background_tasks == 0
    _done(thread)
    assert not [e for e in await _collect(events) if isinstance(e, ToolStarted)]


def _usage_update(thread: ChatThread, total_in: int, last_in: int) -> None:
    _say(
        thread,
        "thread/tokenUsage/updated",
        tokenUsage={
            "total": {"inputTokens": total_in, "outputTokens": 0},
            "last": {"inputTokens": last_in, "outputTokens": 0},
            "modelContextWindow": 200_000,
        },
    )


async def test_a_turn_counts_only_its_own_tokens_on_a_resumed_thread():
    thread = _thread(FakeRuntime())  # opened afresh: the thread's history is not known
    events = await thread.start([], _ctx(), {})
    _usage_update(thread, total_in=50_000, last_in=1_000)  # 49k from before this process
    _usage_update(thread, total_in=51_500, last_in=1_500)
    _done(thread)
    assert (await _collect(events))[-1].usage.input_tokens == 2_500
    second = await thread.start([], _ctx(), {})
    _usage_update(thread, total_in=52_000, last_in=500)
    _done(thread)
    assert (await _collect(second))[-1].usage.input_tokens == 500


async def test_a_used_up_plan_says_when_it_resets():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    _done(thread, "failed", {"message": "limit", "codexErrorInfo": "usageLimitExceeded"})
    ended = (await _collect(events))[-1]
    assert ended.error == HarnessError(kind="rate_limited", retry_at=runtime.reset)


@pytest.mark.parametrize(
    ("info", "kind"),
    [
        ("contextWindowExceeded", "context_full"),
        ("unauthorized", "login"),
        ({"httpConnectionFailed": {"httpStatusCode": 401}}, "login"),
        ({"responseStreamDisconnected": {"httpStatusCode": 503}}, "unavailable"),
        ("serverOverloaded", "unavailable"),
        ("sessionBudgetExceeded", "step_limit"),
        ("cyberPolicy", "internal"),
    ],
)
async def test_codex_errors_map_onto_aug_kinds(info, kind):
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    _done(thread, "failed", {"message": "x", "codexErrorInfo": info})
    ended = (await _collect(events))[-1]
    assert ended.error is not None and ended.error.kind == kind


def test_a_codex_login_error_names_codex_login():
    assert "/codex_login" in error_text(HarnessError(kind="login", login_with="codex"))
    assert "/claude_login" not in error_text(HarnessError(kind="login", login_with="codex"))


async def test_an_interrupted_turn_is_cancelled_not_failed():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    assert await thread.interrupt()
    assert runtime.sent("turn/interrupt") == [{"threadId": THREAD, "turnId": "turn-1"}]
    _done(thread, "interrupted")
    ended = (await _collect(events))[-1]
    assert ended.cancelled and ended.error is None


async def test_an_interrupt_codex_refuses_still_ends_the_turn():
    runtime = FakeRuntime()
    runtime.answers["turn/interrupt"] = CodexRpcError(-32600, "no")
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    await thread.interrupt()
    ended = (await _collect(events))[-1]
    assert ended.cancelled


async def test_a_stop_codex_never_confirms_ends_the_turn_after_a_while(monkeypatch):
    monkeypatch.setattr(codex_thread, "INTERRUPT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "_WATCH_SECONDS", 0.01)
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    await thread.interrupt()  # accepted, but turn/completed never comes
    ended = (await asyncio.wait_for(_collect(events), 2))[-1]
    assert ended.cancelled


async def test_a_turn_codex_goes_silent_on_ends_as_unavailable(monkeypatch):
    monkeypatch.setattr(codex_thread, "SILENT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "_WATCH_SECONDS", 0.01)
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    ended = (await asyncio.wait_for(_collect(events), 2))[-1]
    assert ended.error == HarnessError(kind="unavailable")
    await _settle()
    assert runtime.sent("turn/interrupt")  # so it stops spending, too
    _done(thread)  # its late word is ignored
    assert not thread.turn_open


async def test_a_long_command_is_not_silence(monkeypatch):
    monkeypatch.setattr(codex_thread, "SILENT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "_WATCH_SECONDS", 0.01)
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    _item(thread, "started", type="commandExecution", id="c1", command="sleep 600")
    await asyncio.sleep(0.15)
    assert thread.turn_open
    _item(thread, "completed", type="commandExecution", id="c1", command="sleep 600")
    _done(thread)
    ended = (await _collect(events))[-1]
    assert ended.error is None


async def test_a_reader_that_stops_early_interrupts_the_turn():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    events = await thread.start([], _ctx(), {})
    _say(thread, "item/agentMessage/delta", itemId="a", delta="hi")
    await anext(events)
    await events.aclose()
    await _settle()
    assert runtime.sent("turn/interrupt")


async def test_a_dead_process_ends_the_turn_as_unavailable():
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    thread.on_closed()
    ended = (await _collect(events))[-1]
    assert ended.error == HarnessError(kind="unavailable")


async def test_a_refused_turn_start_ends_at_once():
    runtime = FakeRuntime()
    runtime.answers["turn/start"] = CodexRpcError(-32600, "thread not loaded")
    thread = _thread(runtime)
    ended = (await _collect(await thread.start([], _ctx(), {})))[-1]
    assert ended.error is not None and ended.error.kind == "internal"
    assert not thread.turn_open


async def test_a_turn_start_that_breaks_unexpectedly_ends_the_turn():
    runtime = FakeRuntime()
    runtime.answers["turn/start"] = KeyError("surprise")
    thread = _thread(runtime)
    ended = (await _collect(await thread.start([], _ctx(), {})))[-1]
    assert ended.error == HarnessError(kind="internal")
    assert not thread.turn_open


async def test_steer_reaches_only_an_attended_turn_in_flight():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    assert not await thread.steer([{"type": "text", "text": "x"}])
    events = await thread.start([], _ctx(), {})
    assert await thread.steer([{"type": "text", "text": "also"}])
    assert runtime.sent("turn/steer")[-1]["expectedTurnId"] == "turn-1"
    runtime.answers["turn/steer"] = CodexRpcError(-32600, "not steerable")
    assert not await thread.steer([{"type": "text", "text": "again"}])
    _done(thread)
    await _collect(events)


async def test_a_turn_codex_starts_by_itself_is_delivered():
    delivered: list = []
    thread = _thread(FakeRuntime(), delivered)
    _say(thread, "turn/started", turn={"id": "self-1"})
    assert thread.turn_open
    _say(thread, "item/agentMessage/delta", itemId="a", delta="the research is back")
    _say(thread, "turn/completed", turn={"id": "self-1", "status": "completed"})
    await _settle()
    (events,) = delivered
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["the research is back"]
    assert isinstance(events[-1], TurnEnded)


async def test_a_user_turn_takes_over_one_codex_started():
    delivered: list = []
    runtime = FakeRuntime()
    runtime.answers["turn/start"] = {"turn": {"id": "self-1"}}  # Codex joins it
    thread = _thread(runtime, delivered)
    _say(thread, "turn/started", turn={"id": "self-1"})
    events = await thread.start([{"type": "text", "text": "hi"}], _ctx(), {})
    _say(thread, "item/agentMessage/delta", itemId="a", delta="for you")
    _say(thread, "turn/completed", turn={"id": "self-1", "status": "completed"})
    collected = await _collect(events)
    assert [e.text for e in collected if isinstance(e, TextDelta)] == ["for you"]
    await _settle()
    (nothing_said,) = delivered
    assert [type(e) for e in nothing_said] == [TurnEnded]


async def test_compaction_says_so_and_forgets_the_context():
    runtime = FakeRuntime()
    thread = _thread(runtime)
    _say(
        thread,
        "thread/tokenUsage/updated",
        tokenUsage={
            "total": {"inputTokens": 10, "outputTokens": 1},
            "last": {"inputTokens": 50_000, "outputTokens": 500},
            "modelContextWindow": 200_000,
        },
    )
    assert thread.context is not None and thread.context.used == 50_500
    events = await thread.compact(_ctx())
    assert runtime.sent("thread/compact/start") == [{"threadId": THREAD}]
    _say(thread, "turn/started", turn={"id": "compact-1"})
    _item(thread, "completed", type="contextCompaction", id="cc")
    _say(thread, "turn/completed", turn={"id": "compact-1", "status": "completed"})
    collected = await _collect(events)
    assert any(isinstance(e, StatusUpdate) for e in collected)
    assert thread.context is None


async def test_compaction_ends_a_turn_codex_started_by_itself():
    delivered: list = []
    thread = _thread(FakeRuntime(), delivered)
    _say(thread, "turn/started", turn={"id": "self-1"})
    await thread.compact(_ctx())
    await _settle()
    (events,) = delivered
    assert isinstance(events[-1], TurnEnded)


async def test_a_stop_before_codex_names_the_turn_still_stops_it():
    runtime = FakeRuntime()
    runtime.answers["thread/compact/start"] = {}  # the compaction's turn is named later
    delivered: list = []
    thread = _thread(runtime, delivered)
    events = await thread.compact(_ctx())
    assert await thread.interrupt()
    assert not runtime.sent("turn/interrupt")
    _say(thread, "turn/started", turn={"id": "late-1"})
    await _settle()
    assert runtime.sent("turn/interrupt") == [{"threadId": THREAD, "turnId": "late-1"}]
    _say(thread, "turn/completed", turn={"id": "late-1", "status": "interrupted"})
    ended = (await _collect(events))[-1]
    assert ended.cancelled
    assert not delivered  # never mistaken for a turn Codex started by itself


async def test_a_turn_named_after_it_ended_here_is_stopped_and_ignored(monkeypatch):
    monkeypatch.setattr(codex_thread, "INTERRUPT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "_WATCH_SECONDS", 0.01)
    runtime = FakeRuntime()
    runtime.answers["thread/compact/start"] = {}
    delivered: list = []
    thread = _thread(runtime, delivered)
    events = await thread.compact(_ctx())
    await thread.interrupt()
    await asyncio.wait_for(_collect(events), 2)  # ended here: Codex never confirmed
    _say(thread, "turn/started", turn={"id": "late-1"})
    await _settle()
    assert runtime.sent("turn/interrupt") == [{"threadId": THREAD, "turnId": "late-1"}]
    _say(thread, "item/agentMessage/delta", itemId="a", delta="nobody hears this")
    assert not delivered and not thread.turn_open


async def test_a_turn_ended_here_cannot_end_the_next_one():
    runtime = FakeRuntime()
    runtime.answers["turn/interrupt"] = CodexRpcError(-32600, "no")
    thread = _thread(runtime)
    first = await thread.start([], _ctx(), {})
    await thread.interrupt()  # refused: ended here, Codex may still run turn-1
    await _collect(first)
    runtime.answers["turn/start"] = {"turn": {"id": "turn-2"}}
    second = await thread.start([], _ctx(), {})
    _done(thread, "interrupted")  # turn-1's late word
    _say(thread, "item/agentMessage/delta", itemId="a", delta="still here")
    _say(thread, "turn/completed", turn={"id": "turn-2", "status": "completed"})
    collected = await _collect(second)
    assert [e.text for e in collected if isinstance(e, TextDelta)] == ["still here"]
    assert not collected[-1].cancelled


async def test_what_codex_says_late_about_a_turn_ended_here_never_reaches_the_next():
    runtime = FakeRuntime()
    runtime.answers["turn/interrupt"] = CodexRpcError(-32600, "no")
    thread = _thread(runtime)
    first = await thread.start([], _ctx(), {})
    await thread.interrupt()  # refused: ended here, Codex may still run turn-1
    await _collect(first)
    runtime.answers["turn/start"] = {"turn": {"id": "turn-2"}}
    second = await thread.start([], _ctx(), {})
    _say(thread, "item/agentMessage/delta", itemId="old", delta="from turn 1", turnId="turn-1")
    _say(thread, "error", turnId="turn-1", willRetry=False, error={"codexErrorInfo": "x"})
    late = await thread.on_tool_call(_call("progress", "late", turn_id="turn-1"))
    _say(thread, "item/agentMessage/delta", itemId="new", delta="from turn 2", turnId="turn-2")
    _say(thread, "turn/completed", turn={"id": "turn-2", "status": "completed"})
    collected = await _collect(second)
    assert [e.text for e in collected if isinstance(e, TextDelta)] == ["from turn 2"]
    assert collected[-1].error is None
    assert late["success"] is False  # never ran
    assert not [e for e in collected if isinstance(e, ToolProgress)]


async def test_a_tool_call_after_stop_never_runs():
    ran: list[str] = []

    @tool
    async def remember() -> str:
        """Record that it ran."""
        ran.append("yes")
        return "ran"

    thread = ChatThread(
        "ctx",
        FakeRuntime(),
        tools=lambda: {"remember": remember},
        reply_to=ADDR,
        on_unsolicited=lambda events: asyncio.sleep(0),
    )
    thread.bind(THREAD)
    await thread.start([], _ctx(), {})
    await thread.interrupt()
    answer = await thread.on_tool_call(_call("remember", "after-stop"))
    assert answer["success"] is False and not ran


async def test_a_named_turn_nobody_waits_for_is_stopped_at_once():
    runtime = FakeRuntime()
    answered = asyncio.Event()

    async def slow_request(method: str, params: dict | None = None) -> dict:
        runtime.requests.append((method, params or {}))
        if method == "turn/start":
            await answered.wait()
        return {}

    runtime.request = slow_request  # type: ignore[method-assign]
    thread = _thread(runtime)
    starting = asyncio.create_task(thread.start([], _ctx(), {}))
    await _settle()
    _say(thread, "turn/started", turn={"id": "named"})  # before turn/start answers
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await _settle()
    assert runtime.sent("turn/interrupt") == [{"threadId": THREAD, "turnId": "named"}]
    _say(thread, "item/agentMessage/delta", itemId="a", delta="nobody", turnId="named")
    assert not thread.turn_open


async def test_the_turn_start_answer_says_which_turn_is_ours():
    delivered: list = []
    runtime = FakeRuntime()
    runtime.answers["turn/start"] = {"turn": {"id": "ours"}}
    thread = _thread(runtime, delivered)
    _say(thread, "turn/started", turn={"id": "self-1"})  # Codex started one by itself
    events = await thread.start([], _ctx(), {})
    _say(thread, "turn/completed", turn={"id": "self-1", "status": "completed"})  # not ours
    assert thread.turn_open
    _say(thread, "item/agentMessage/delta", itemId="a", delta="for you", turnId="ours")
    _say(thread, "turn/completed", turn={"id": "ours", "status": "completed"})
    collected = await _collect(events)
    assert [e.text for e in collected if isinstance(e, TextDelta)] == ["for you"]


async def test_a_turn_start_nobody_waits_for_is_stopped_once_codex_names_it():
    runtime = FakeRuntime()
    answered = asyncio.Event()

    async def slow_request(method: str, params: dict | None = None) -> dict:
        runtime.requests.append((method, params or {}))
        if method == "turn/start":
            await answered.wait()
        return {}

    runtime.request = slow_request  # type: ignore[method-assign]
    thread = _thread(runtime)
    starting = asyncio.create_task(thread.start([], _ctx(), {}))
    await _settle()
    starting.cancel()  # the reader went away while turn/start was in flight
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert not thread.turn_open
    _say(thread, "turn/started", turn={"id": "orphan"})
    await _settle()
    assert runtime.sent("turn/interrupt") == [{"threadId": THREAD, "turnId": "orphan"}]


# -- AUG's tools -------------------------------------------------------------------------


async def test_a_tool_runs_with_the_turn_context_and_files_its_events_under_the_call():
    thread = _thread(FakeRuntime())
    events = await thread.start([], _ctx(), {})
    answer = await thread.on_tool_call(_call("progress", "call-9"))
    assert answer == {
        "contentItems": [{"type": "inputText", "text": "done for ctx"}],
        "success": True,
    }
    _done(thread)
    (step,) = [e for e in await _collect(events) if isinstance(e, ToolProgress)]
    assert step.parent_id == "call-9"


async def test_stop_cancels_a_running_tool_and_codex_hears_so():
    thread = _thread(FakeRuntime())
    await thread.start([], _ctx(), {})
    call = asyncio.create_task(thread.on_tool_call(_call("forever", "c")))
    await asyncio.sleep(0.01)
    await thread.interrupt()
    answer = await asyncio.wait_for(call, 2)
    assert answer["success"] is False
    assert "/stop" in answer["contentItems"][0]["text"]


async def test_a_tool_call_codex_stops_waiting_for_is_cancelled():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def watched() -> str:
        """Run until cancelled."""
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    thread = ChatThread(
        "ctx",
        FakeRuntime(),
        tools=lambda: {"watched": watched},
        reply_to=ADDR,
        on_unsolicited=lambda events: asyncio.sleep(0),
    )
    thread.bind(THREAD)
    await thread.start([], _ctx(), {})
    call = asyncio.create_task(thread.on_tool_call(_call("watched", "c")))
    await asyncio.wait_for(started.wait(), 2)
    call.cancel()  # the app-server went away: nobody waits for the answer
    await asyncio.wait_for(cancelled.wait(), 2)


async def test_a_dead_process_cancels_the_running_tools():
    thread = _thread(FakeRuntime())
    await thread.start([], _ctx(), {})
    call = asyncio.create_task(thread.on_tool_call(_call("forever", "c")))
    await asyncio.sleep(0.01)
    thread.on_closed()
    assert (await asyncio.wait_for(call, 2))["success"] is False


async def test_a_tool_that_is_gone_says_so():
    answer = await run_tool(TOOLS, "vanished", {}, "c", None)
    assert answer["success"] is False
    assert "no longer available" in answer["contentItems"][0]["text"]


def test_tool_outcomes_become_codex_content():
    outcome = ToolOutcome(blocks=("hi", ToolImage(media_type="image/png", data="AAAA")))
    assert tool_answer(outcome) == {
        "contentItems": [
            {"type": "inputText", "text": "hi"},
            {"type": "inputImage", "imageUrl": "data:image/png;base64,AAAA"},
        ],
        "success": True,
    }


def test_tools_not_always_loaded_wait_in_the_aug_namespace():
    specs = dynamic_tools([progress, forever], frozenset({"progress"}))
    assert specs[0]["name"] == "progress" and "deferLoading" not in specs[0]
    assert specs[1]["type"] == "namespace" and specs[1]["name"] == "aug"
    assert [(t["name"], t["deferLoading"]) for t in specs[1]["tools"]] == [("forever", True)]


@pytest.mark.parametrize(
    ("ran", "written"),
    [
        (shlex.join(["/bin/bash", "-lc", hushed_command("echo hi")]), "echo hi"),
        (shlex.join(["/bin/sh", "-c", hushed_command("echo 'a b'")]), "echo 'a b'"),
        ("/bin/zsh -lc 'ls -la'", "ls -la"),
        ("plain", "plain"),
    ],
)
def test_the_command_shown_is_the_one_the_agent_wrote(ran, written):
    assert original_command(ran) == written


# -- guard -------------------------------------------------------------------------------


def test_the_guard_runs_shell_commands_through_hushed():
    decision = guard.decide({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    output = decision["hookSpecificOutput"]
    assert output["permissionDecision"] == "allow"
    assert output["updatedInput"] == {"command": hushed_command("ls")}


@pytest.mark.parametrize(
    "call",
    [
        {"tool_name": "Bash", "tool_input": {"command": f"cat {DATA_DIR}/settings.json"}},
        {
            "tool_name": "apply_patch",
            "tool_input": {"command": f"*** Update File: {DATA_DIR}/keys/x"},
        },
        {"tool_name": "view_image", "tool_input": {"path": f"{CODEX_DIR}/auth.json"}},
    ],
)
def test_the_guard_refuses_blocked_paths(call):
    output = guard.decide(call)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny" and output["permissionDecisionReason"]


@pytest.mark.parametrize(
    "call",
    [
        {"tool_name": "Bash", "tool_input": {"command": ["cat", "/etc/passwd"]}},
        {"tool_name": "Bash", "tool_input": "ls"},
        {"tool_name": "Bash"},
        {"tool_name": "apply_patch", "tool_input": {"patch": "*** Add File: x"}},
        {"tool_name": "view_image", "tool_input": {"path": None}},
        {"tool_name": "something_new", "tool_input": {"command": "ls"}},
    ],
)
def test_the_guard_refuses_a_call_it_does_not_know(call):
    assert guard.decide(call)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_guard_lets_other_calls_be():
    assert guard.decide({"tool_name": "apply_patch", "tool_input": {"command": "*** x.txt"}}) == {}
    assert guard.decide({"tool_name": "view_image", "tool_input": {"path": "/tmp/a.png"}}) == {}


def test_the_guard_resolves_relative_patch_and_image_paths():
    cwd = str(DATA_DIR / "workspace")
    patch_call = {
        "tool_name": "apply_patch",
        "cwd": cwd,
        "tool_input": {"command": "*** Begin Patch\n*** Update File: ../codex/auth.json\n"},
    }
    image_call = {"tool_name": "view_image", "cwd": cwd, "tool_input": {"path": "../keys/id"}}
    for call in (patch_call, image_call):
        assert guard.decide(call)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_guard_reads_patch_headers_as_leniently_as_codex():
    patch_call = {
        "tool_name": "apply_patch",
        "cwd": str(DATA_DIR / "workspace"),
        "tool_input": {"command": "*** Begin Patch\n  ***  Update File:  ../codex/auth.json \n"},
    }
    assert guard.decide(patch_call)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_guard_does_not_wrap_a_command_twice():
    once = hushed_command("ls")
    assert guard.decide({"tool_name": "Bash", "tool_input": {"command": once}}) == {}


def test_a_broken_guard_refuses_the_call(monkeypatch, capsys):
    def boom(call: dict) -> dict:
        raise ValueError("corrupt settings")

    monkeypatch.setattr(guard, "decide", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    guard.main()
    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"


def _run_hook(command: str, call: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", command], input=json.dumps(call), capture_output=True, text=True
    )


def test_the_guard_runs_as_a_hook_process():
    result = _run_hook(
        guard.hook_command(), {"tool_name": "Bash", "tool_input": {"command": "pwd"}}
    )
    assert result.returncode == 0
    command = json.loads(result.stdout)["hookSpecificOutput"]["updatedInput"]["command"]
    assert command.startswith("hushed run")


def test_a_guard_process_that_fails_refuses_the_call(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    result = _run_hook(guard.hook_command(), {"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert result.returncode == 2  # Codex refuses the call
    assert "refused" in result.stderr


def test_auth_json_is_a_blocked_path():
    assert path_block_reason(CODEX_DIR / "auth.json")


# -- runtime -----------------------------------------------------------------------------


def test_usage_windows_are_named_and_reset_ones_dropped(tmp_path):
    runtime = CodexRuntime(tmp_path)
    soon = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
    past = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
    runtime._on_notification(
        "account/rateLimits/updated",
        {
            "rateLimits": {
                "limitId": "codex",
                "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": soon},
                "secondary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": past},
            }
        },
    )
    (limit,) = runtime.limits
    assert (limit.window, limit.utilization) == ("five_hour", 1.0)
    assert runtime.retry_at() == datetime.fromtimestamp(soon, UTC)


def test_codex_runs_with_aug_settings_only(tmp_path):
    runtime = CodexRuntime(tmp_path)
    (tmp_path / "managed_config.toml").write_text("[hooks]\n")  # would outrank AUG's
    (tmp_path / "requirements.toml").write_text("")
    runtime._write_config()
    assert not (tmp_path / "managed_config.toml").exists()
    assert not (tmp_path / "requirements.toml").exists()
    assert json.dumps(guard.hook_command()) in (tmp_path / "config.toml").read_text()
    assert runtime._config["features.multi_agent"] == "false"  # v1 reaches other threads


async def test_letting_go_of_a_thread_never_starts_codex(tmp_path):
    runtime = CodexRuntime(tmp_path)
    runtime.mark_loaded(THREAD)
    listener = MagicMock(context_id="ctx")
    runtime.listen(THREAD, listener)
    runtime.unload(THREAD, listener)  # no server running: nothing to do, nothing started
    assert runtime._server is None and not runtime.is_loaded(THREAD)
    assert THREAD not in runtime._listeners


def test_a_thread_is_only_let_go_of_by_whoever_listens_to_it(tmp_path):
    runtime = CodexRuntime(tmp_path)
    old, reopened = MagicMock(context_id="ctx"), MagicMock(context_id="ctx")
    runtime.listen(THREAD, old)
    assert runtime.listen(THREAD, reopened)  # the same chat, opened again
    runtime.mark_loaded(THREAD)
    runtime.forget(THREAD, old)
    runtime.unload(THREAD, old)
    assert runtime._listeners[THREAD] is reopened and THREAD in runtime._loaded


async def test_a_resume_waits_for_the_let_go_before_it(tmp_path):
    runtime = CodexRuntime(tmp_path)
    sent: list[str] = []
    release = asyncio.Event()

    class Server:
        alive = True

        async def request(self, method: str, params: dict | None = None) -> dict:
            if method == "thread/unsubscribe":
                await release.wait()
            sent.append(method)
            return {}

    runtime._server = Server()  # type: ignore[assignment]
    listener = MagicMock(context_id="ctx")
    runtime.listen(THREAD, listener)
    runtime.unload(THREAD, listener)
    waiting = asyncio.create_task(runtime.unloaded(THREAD))
    await _settle()
    assert not waiting.done()
    release.set()
    await asyncio.wait_for(waiting, 2)
    assert sent == ["thread/unsubscribe"]


def test_a_chat_never_takes_over_another_chats_thread(tmp_path):
    runtime = CodexRuntime(tmp_path)
    mine = MagicMock(context_id="mine")
    runtime.listen(THREAD, mine)
    assert not runtime.listen(THREAD, MagicMock(context_id="theirs"))
    assert runtime._listeners[THREAD] is mine


def test_notifications_reach_the_thread_they_are_about(tmp_path):
    runtime = CodexRuntime(tmp_path)
    listener = MagicMock()
    runtime.listen(THREAD, listener)
    runtime._on_notification("item/started", {"threadId": THREAD, "item": {}})
    runtime._on_notification("item/started", {"threadId": "other", "item": {}})
    listener.on_notification.assert_called_once_with(
        "item/started", {"threadId": THREAD, "item": {}}
    )


async def test_codex_requests_aug_does_not_answer_are_refused(tmp_path):
    runtime = CodexRuntime(tmp_path)
    with pytest.raises(CodexRpcError):
        await runtime._on_request("item/commandExecution/requestApproval", {})
    assert await runtime._on_request("item/tool/requestUserInput", {}) == {"answers": {}}


async def test_a_finished_sign_in_resolves_its_wait_and_is_forgotten(tmp_path):
    runtime = CodexRuntime(tmp_path)
    signed_in, refused = runtime.login_finished("L1"), runtime.login_finished("L2")
    runtime._on_notification("account/login/completed", {"loginId": "L1", "success": True})
    runtime._on_notification(
        "account/login/completed", {"loginId": "L2", "success": False, "error": "denied"}
    )
    runtime._on_notification("account/login/completed", {"loginId": "L3", "success": True})
    assert (await signed_in, await refused) == (None, "denied")
    await asyncio.sleep(0)
    assert not runtime._logins


class _StuckServer:
    """An app-server that never answers."""

    alive = True

    def __init__(self) -> None:
        self.closed = False

    async def request(self, method: str, params: dict | None = None) -> dict:
        raise TimeoutError

    async def close(self) -> None:
        self.closed = True
        self.alive = False


async def test_a_stuck_app_server_is_stopped_and_its_turns_end(tmp_path):
    runtime = CodexRuntime(tmp_path)
    stuck = _StuckServer()
    runtime._server = stuck  # type: ignore[assignment]
    listener = MagicMock()
    runtime.listen(THREAD, listener)
    with pytest.raises(TimeoutError):
        await runtime.request("turn/start", {})
    assert stuck.closed and runtime._server is None  # the next request starts a new one
    listener.on_closed.assert_called_once()


async def test_closing_the_runtime_ends_what_is_in_flight(tmp_path):
    runtime = CodexRuntime(tmp_path)
    runtime._server = _StuckServer()  # type: ignore[assignment]
    listener = MagicMock()
    runtime.listen(THREAD, listener)
    waiting = runtime.login_finished("L1")
    await runtime.close()
    listener.on_closed.assert_called_once()
    assert await waiting == "the Codex process exited"


class _HookServer:
    """An app-server whose hooks/list says what *trusted* says, recording writes."""

    def __init__(self, trusted: list[bool]) -> None:
        self.trusted = trusted
        self.writes: list[dict] = []

    async def request(self, method: str, params: dict | None = None) -> dict:
        if method == "config/value/write":
            self.writes.append(params or {})
            return {}
        status = "trusted" if self.trusted.pop(0) else "untrusted"
        hook = {
            "command": guard.hook_command(),
            "key": "k",
            "currentHash": "h",
            "trustStatus": status,
        }
        return {"data": [{"hooks": [hook]}]}


async def test_the_guard_hook_is_trusted_and_checked_again(tmp_path):
    server = _HookServer(trusted=[False, True])
    await CodexRuntime(tmp_path)._trust_guard(server)  # type: ignore[arg-type]
    assert [w["value"] for w in server.writes] == ["h"]


async def test_a_server_that_will_not_trust_the_guard_is_not_used(tmp_path):
    server = _HookServer(trusted=[False, False])
    with pytest.raises(CodexUnusable, match="guard"):
        await CodexRuntime(tmp_path)._trust_guard(server)  # type: ignore[arg-type]


# -- harness -----------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch) -> CodexHarness:
    monkeypatch.setattr("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(codex_harness, "session_prompt", lambda *args, **kwargs: "the prompt")
    run_registry.clear()
    profile = CodexProfile(model="m", effort="low", tools=[])
    return CodexHarness(profile, runtime=FakeRuntime(tmp_path))  # type: ignore[arg-type]


def _user(text: str) -> Message:
    return Message(parts=[TextPart(text=text)], sender=Sender(interface="t", user_id="u"))


async def _run(harness: CodexHarness, text: str) -> list:
    async with run_registry.run_lock("ctx"):
        return [
            e
            async for e in harness.run(
                _user(text), "ctx", TurnContext(reply_to=ADDR, attended=True)
            )
        ]


class _Refusing:
    """A thread whose turn takes no steering."""

    async def start(self, items: list, ctx: ToolContext, params: dict) -> Any:
        async def events() -> Any:
            yield TurnEnded()

        return events()

    async def steer(self, items: list) -> bool:
        return False


async def test_messages_injected_early_that_the_turn_refuses_come_back(harness, monkeypatch):
    early = _user("also")

    async def thread(context_id: str, reply_to: Address) -> Any:
        await harness.inject(early, "ctx")  # arrives while the thread is being opened
        return _Refusing()

    monkeypatch.setattr(harness, "_thread", thread)
    events = await _run(harness, "hi")
    assert events[-1].unconsumed == [early]


async def test_codex_that_will_not_start_is_unavailable_not_internal(harness, monkeypatch):
    async def broken() -> bool:
        raise OSError("no binary")

    monkeypatch.setattr(harness, "_signed_in", broken)
    events = await _run(harness, "hi")
    assert events[-1].error == HarnessError(kind="unavailable")


async def test_codex_refusing_the_guard_is_unavailable_and_keeps_early_messages(
    harness, monkeypatch
):
    early = _user("also")

    async def no_guard(context_id: str, reply_to: Address) -> Any:
        await harness.inject(early, "ctx")
        raise CodexUnusable("codex does not run AUG's guard hook")

    monkeypatch.setattr(harness, "_thread", no_guard)
    ended = (await _run(harness, "hi"))[-1]
    assert ended.error == HarnessError(kind="unavailable")
    assert ended.unconsumed == [early]


async def test_an_unexpected_failure_opening_the_thread_is_internal(harness, monkeypatch):
    async def broken(context_id: str, reply_to: Address) -> Any:
        raise KeyError("thread")

    monkeypatch.setattr(harness, "_thread", broken)
    ended = (await _run(harness, "hi"))[-1]
    assert ended.error == HarnessError(kind="internal")


async def test_stop_reaches_a_turn_no_run_owns(harness):
    thread = await harness._thread("ctx", ADDR)
    events = await thread.compact(_ctx())
    assert await harness.cancel("ctx")
    _say(thread, "turn/started", turn={"id": "compact-1"})
    _say(thread, "turn/completed", turn={"id": "compact-1", "status": "interrupted"})
    assert (await _collect(events))[-1].cancelled
    assert not await harness.cancel("ctx")  # nothing left to stop


async def test_a_compaction_that_takes_too_long_is_stopped(harness, monkeypatch):
    await harness._thread("ctx", ADDR)  # a thread to compact
    monkeypatch.setattr(codex_harness, "COMPACT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "INTERRUPT_SECONDS", 0.05)
    monkeypatch.setattr(codex_thread, "_WATCH_SECONDS", 0.01)
    with pytest.raises(ValueError, match="too long"):
        await harness.compact("ctx")
    await asyncio.sleep(0.2)  # stopped, though Codex never says so
    assert not harness._threads["ctx"].turn_open


async def test_a_stopped_compaction_says_so(harness):
    thread = await harness._thread("ctx", ADDR)
    compaction = asyncio.create_task(harness.compact("ctx"))
    await _until(lambda: thread.turn_open)
    await thread.interrupt()
    _say(thread, "turn/started", turn={"id": "compact-1"})
    _say(thread, "turn/completed", turn={"id": "compact-1", "status": "interrupted"})
    with pytest.raises(ValueError, match="stopped"):
        await compaction


async def test_a_oneshot_that_takes_too_long_is_interrupted(harness, monkeypatch):
    monkeypatch.setattr(codex_harness, "_ONESHOT_SECONDS", 0.05)
    runtime = harness._runtime
    runtime.answers["thread/start"] = {"thread": {"id": "one"}}  # type: ignore[attr-defined]
    with pytest.raises(TimeoutError):
        await harness.oneshot("hi")
    assert runtime.sent("turn/interrupt") == [{"threadId": "one", "turnId": "turn-1"}]  # type: ignore[attr-defined]
    assert runtime.sent("thread/unsubscribe") == [{"threadId": "one"}]  # type: ignore[attr-defined]


async def test_idle_threads_past_the_limit_are_let_go(harness, monkeypatch):
    monkeypatch.setattr(codex_harness, "MAX_OPEN_THREADS", 1)
    runtime = harness._runtime
    for n in range(3):
        runtime.answers["thread/start"] = {"thread": {"id": f"t{n}"}}  # type: ignore[attr-defined]
        await harness._thread(f"ctx{n}", ADDR)
    assert set(harness._threads) == {"ctx1", "ctx2"}  # the newest, and one idle one
    assert runtime.sent("thread/unsubscribe") == [{"threadId": "t0"}]  # type: ignore[attr-defined]
    await harness._thread("ctx0", ADDR)  # resumes
    assert runtime.sent("thread/resume")[-1]["threadId"] == "t0"  # type: ignore[attr-defined]


async def test_a_busy_thread_is_never_let_go(harness, monkeypatch):
    monkeypatch.setattr(codex_harness, "MAX_OPEN_THREADS", 0)
    runtime = harness._runtime
    busy = await harness._thread("busy", ADDR)
    await busy.start([], _ctx(), {})
    runtime.answers["thread/start"] = {"thread": {"id": "other-thread"}}  # type: ignore[attr-defined]
    await harness._thread("other", ADDR)
    await harness._thread("third", ADDR)  # lets go of idle ones
    assert "busy" in harness._threads


async def test_a_thread_being_opened_is_never_let_go(harness, monkeypatch):
    monkeypatch.setattr(codex_harness, "MAX_OPEN_THREADS", 0)
    runtime = harness._runtime
    opening = await harness._thread("opening", ADDR)
    async with harness._opening["opening"]:  # another turn is opening it right now
        runtime.answers["thread/start"] = {"thread": {"id": "t2"}}  # type: ignore[attr-defined]
        await harness._thread("other", ADDR)
    assert harness._threads["opening"] is opening
