"""The Harness contract, and LangGraphHarness on a real graph with a scripted model."""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import psycopg
import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from aug.api.interfaces.base import IncomingMessage, TextContent
from aug.api.interfaces.rest import RestApiInterface
from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent
from aug.core.harness import (
    Address,
    ApprovalDecision,
    ApprovalRequested,
    ApprovalResolved,
    FileEmitted,
    FilePart,
    HarnessError,
    Message,
    ModelTier,
    Sender,
    StatusUpdate,
    TextDelta,
    TextPart,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnContext,
    TurnEnded,
    TurnStarted,
    error_text,
    pending_approvals,
)
from aug.core.harness.langgraph import (
    ATTENDED_MAX_TOOL_ROUNDS,
    UNATTENDED_MAX_TOOL_ROUNDS,
    LangGraphHarness,
    _harness_error,
)
from aug.core.harness.langgraph_events import parse_event
from aug.core.prompts import (
    HARNESS_ERROR_MESSAGES,
    HARNESS_STEP_LIMIT_NOTE,
    HARNESS_STOPPED_MESSAGE,
    HARNESS_TOOL_STOPPED_RESULT,
    HARNESS_TURN_STOPPED_NOTE,
    MID_RUN_INJECTION_PREFIX,
)
from aug.core.reflexes import ReflexOutput
from aug.core.run import AGENT_RUN_CONFIG_KEY, AgentRun, run_registry
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.approval import requires_approval
from tests.harness_fakes import ScriptedHarness

TURN = TurnContext(
    reply_to=Address(interface="telegram", conversation_id="tg-1", chat_id="1"), attended=True
)
SENDER = Sender(interface="telegram", user_id="1")


def _msg(text: str) -> Message:
    return Message(parts=[TextPart(text=text)], sender=SENDER)


async def _collect(stream: AsyncIterator) -> list:
    return [event async for event in stream]


@pytest.fixture(autouse=True)
def transcripts(tmp_path):
    with patch("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts"):
        yield tmp_path / "transcripts"


@pytest.fixture(autouse=True)
def clean_runs():
    run_registry.clear()
    pending_approvals.clear()
    yield
    run_registry.clear()
    pending_approvals.clear()


# ---------------------------------------------------------------------------
# run() contract — any harness
# ---------------------------------------------------------------------------


class _Crashing(ScriptedHarness):
    async def _run(self, message, context_id, turn):
        yield TextDelta(text="partial")
        raise RuntimeError("boom")


async def test_run_starts_with_turn_started_and_ends_with_turn_ended() -> None:
    events = await _collect(ScriptedHarness([TextDelta(text="hi")]).run(_msg("x"), "c1", TURN))

    assert events[0] == TurnStarted(context_id="c1")
    assert events[1] == TextDelta(text="hi")
    assert events[-1] == TurnEnded()


async def test_an_exception_inside_run_becomes_turn_ended_internal() -> None:
    events = await _collect(_Crashing().run(_msg("x"), "c1", TURN))

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].error == HarnessError(kind="internal")


async def test_the_harness_turn_ended_is_passed_through_once() -> None:
    ended = TurnEnded(error=HarnessError(kind="rate_limited"))
    harness = ScriptedHarness([ended, TextDelta(text="never")])

    events = await _collect(harness.run(_msg("x"), "c1", TURN))

    assert [e for e in events if isinstance(e, TurnEnded)] == [ended]
    assert TextDelta(text="never") not in events


async def test_turn_is_written_to_the_conversation_transcript(transcripts) -> None:
    harness = ScriptedHarness(
        [
            ToolStarted(id="r1", name="note", args={"text": "x"}),
            TextDelta(text="done"),
        ]
    )
    await _collect(harness.run(_msg("hello"), "tg-1-0", TURN))

    records = [json.loads(line) for line in (transcripts / "tg-1.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["turn_started", "tool_started", "turn_ended"]
    assert records[0]["text"] == "hello"
    assert records[1]["tool"] == "note"
    assert records[2]["text"] == "done"
    assert {r["context_id"] for r in records} == {"tg-1-0"}


async def test_transcript_logs_files_both_ways_without_their_bytes(transcripts, tmp_path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg bytes")
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png bytes")
    sent = FileEmitted(
        file_id="0123456789abcdef", path=chart, media_type="image/png", filename="chart.png"
    )
    message = Message(
        parts=[FilePart(path=photo, media_type="image/jpeg", filename="photo.jpg")],
        sender=SENDER,
    )

    await _collect(ScriptedHarness([sent]).run(message, "tg-1-0", TURN))

    raw = (transcripts / "tg-1.jsonl").read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    [received] = records[0]["files"]
    file_sent = next(r for r in records if r["event"] == "file_sent")
    assert received == {
        "path": str(photo),
        "media_type": "image/jpeg",
        "filename": "photo.jpg",
        "size": 10,
        "sha256": hashlib.sha256(b"jpeg bytes").hexdigest(),
    }
    assert (file_sent["filename"], file_sent["size"]) == ("chart.png", 9)
    assert "jpeg bytes" not in raw and "png bytes" not in raw


async def test_a_failing_transcript_never_fails_the_turn(transcripts) -> None:
    transcripts.parent.mkdir(exist_ok=True)
    transcripts.write_text("a file where the directory should be")

    events = await _collect(ScriptedHarness([TextDelta(text="hi")]).run(_msg("x"), "c1", TURN))

    assert events[-1] == TurnEnded()


def test_error_text_per_kind() -> None:
    for kind, text in HARNESS_ERROR_MESSAGES.items():
        assert error_text(HarnessError(kind=kind)) == text


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _status_error(cls, status: int, code: str | None = None):
    request = httpx.Request("POST", "http://llm/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("fake", response=response, body={"code": code} if code else None)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_status_error(openai.RateLimitError, 429), "rate_limited"),
        (_status_error(openai.BadRequestError, 400, "context_length_exceeded"), "context_full"),
        (ContextOverflowError("too long"), "context_full"),
        (_status_error(openai.AuthenticationError, 401), "auth"),
        (_status_error(openai.PermissionDeniedError, 403), "auth"),
        (_status_error(openai.InternalServerError, 503), "unavailable"),
        (openai.APIConnectionError(request=httpx.Request("POST", "http://llm")), "unavailable"),
        (psycopg.OperationalError("db gone"), "unavailable"),
        (GraphRecursionError("loop"), "step_limit"),
        (_status_error(openai.BadRequestError, 400, "invalid_value"), "internal"),
        (KeyError("x"), "internal"),
    ],
)
def test_harness_error_mapping(exc: Exception, kind: str) -> None:
    assert _harness_error(exc).kind == kind


def test_rate_limit_is_no_longer_reported_as_context_full() -> None:
    error = _harness_error(_status_error(openai.RateLimitError, 429))
    assert "Context window is full" not in error_text(error)


# ---------------------------------------------------------------------------
# LangGraphHarness on a real graph
# ---------------------------------------------------------------------------


@tool
async def slow(seconds: float, config: RunnableConfig) -> str:
    """Wait, stopping early when the run is asked to stop."""
    run = config["configurable"][AGENT_RUN_CONFIG_KEY]
    for _ in range(int(seconds / 0.01)):
        if run.user_requested_stop.is_set():
            return "stopped"
        await asyncio.sleep(0.01)
    return "waited"


stubborn_ran: list[float] = []


@tool
async def stubborn(seconds: float) -> str:
    """Wait, ignoring the stop flag."""
    await asyncio.sleep(seconds)
    stubborn_ran.append(seconds)
    return "waited"


@tool
def stubborn_sync(seconds: float) -> str:
    """Block a worker thread, ignoring the stop flag."""
    time.sleep(seconds)
    return "waited"


@tool
@requires_approval(describe=lambda command: ("box", command))
async def guarded(command: str) -> str:
    """Run a command that needs approval."""
    return f"ran {command}"


class _ScriptAgent(BaseAgent):
    """Replies from a fixed list of AIMessages; records what each model call saw."""

    def __init__(self, replies: list[AIMessage | Exception]) -> None:
        super().__init__()
        self.tools = [slow, stubborn, stubborn_sync, guarded]
        self.replies = list(replies)
        self.seen: list[list] = []

    async def respond(self, state: AgentState) -> AgentStateUpdate:
        self.seen.append(list(state.messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AgentStateUpdate(messages=[reply])


def _call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _harness(*replies) -> tuple[LangGraphHarness, _ScriptAgent]:
    agent = _ScriptAgent(list(replies))
    return LangGraphHarness(agent, MemorySaver()), agent


async def test_inject_is_false_when_nothing_runs() -> None:
    harness, _ = _harness()
    assert await harness.inject(_msg("hi"), "c1") is False
    assert await harness.cancel("c1") is False


async def test_plain_turn_ends_cleanly() -> None:
    harness, agent = _harness(AIMessage(content="hello"))

    events = await _collect(harness.run(_msg("hi"), "c1", TURN))

    assert events[-1] == TurnEnded()
    assert agent.seen[0][-1].content == "hi"
    assert run_registry.get("c1") is None  # the turn closed itself


async def test_injection_lands_at_the_next_tool_round() -> None:
    harness, agent = _harness(_call("slow", {"seconds": 0.3}), AIMessage(content="ok"))

    async def inject_soon() -> bool:
        await asyncio.sleep(0.1)
        return await harness.inject(_msg("also this"), "c1")

    events, injected = await asyncio.gather(
        _collect(harness.run(_msg("start"), "c1", TURN)), inject_soon()
    )

    assert injected is True
    assert agent.seen[1][-1].content == MID_RUN_INJECTION_PREFIX + "also this"
    assert events[-1].unconsumed == []


async def test_a_late_injection_comes_back_unconsumed() -> None:
    harness, _ = _harness(AIMessage(content="final"))
    stream = harness.run(_msg("start"), "c1", TURN)
    assert isinstance(await anext(stream), TurnStarted)
    assert await harness.inject(_msg("too late"), "c1") is True  # turn is open, no pause left

    events = await _collect(stream)

    assert events[-1].unconsumed == [_msg("too late")]


async def test_cancel_stops_the_running_tool() -> None:
    harness, _ = _harness(_call("slow", {"seconds": 5}), AIMessage(content="never"))

    async def cancel_soon() -> bool:
        await asyncio.sleep(0.1)
        return await harness.cancel("c1")

    events, cancelled = await asyncio.wait_for(
        asyncio.gather(_collect(harness.run(_msg("go"), "c1", TURN)), cancel_soon()), 3
    )

    assert cancelled is True
    assert events[-1].cancelled is True


@pytest.mark.parametrize("tool_name", ["stubborn", "stubborn_sync"])
async def test_cancel_does_not_wait_for_a_tool_that_ignores_it(tool_name: str) -> None:
    harness, agent = _harness(_call(tool_name, {"seconds": 2}), AIMessage(content="next turn"))

    async def cancel_soon() -> bool:
        await asyncio.sleep(0.1)
        return await harness.cancel("c1")

    started = time.monotonic()
    events, _ = await asyncio.wait_for(
        asyncio.gather(_collect(harness.run(_msg("go"), "c1", TURN)), cancel_soon()), 3
    )

    assert time.monotonic() - started < 1
    assert events[-1] == TurnEnded(cancelled=True)
    # The cut-off tool call doesn't break the session: the next turn runs normally.
    after = await _collect(harness.run(_msg("again"), "c1", TURN))
    assert after[-1] == TurnEnded()
    assert agent.seen[-1][-1].content == "again"


async def test_a_consumer_that_goes_away_takes_the_running_tool_with_it() -> None:
    """A REST client hanging up cancels the task reading the turn; the tool must stop
    too, since the run lock is released and /stop can no longer reach it."""
    stubborn_ran.clear()
    harness, _ = _harness(_call("stubborn", {"seconds": 0.5}), AIMessage(content="never"))
    finished: list[str] = []

    async def consume() -> None:
        async for event in harness.run(_msg("go"), "c1", TURN):
            if isinstance(event, ToolEnded):
                finished.append(event.tool_name)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.7)

    assert finished == []
    assert stubborn_ran == []


async def test_the_next_turn_knows_the_user_stopped_the_tool() -> None:
    harness, agent = _harness(_call("stubborn", {"seconds": 2}), AIMessage(content="next turn"))

    async def cancel_soon() -> bool:
        await asyncio.sleep(0.1)
        return await harness.cancel("c1")

    await asyncio.gather(_collect(harness.run(_msg("go"), "c1", TURN)), cancel_soon())
    await _collect(harness.run(_msg("what happened?"), "c1", TURN))

    *_, call, result, question = agent.seen[-1]
    assert call.tool_calls[0]["name"] == "stubborn"
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "c1"
    assert result.content == HARNESS_TOOL_STOPPED_RESULT
    assert question.content == "what happened?"


async def test_the_next_turn_knows_the_user_stopped_the_reply() -> None:
    class _SlowModel(_ScriptAgent):
        async def respond(self, state: AgentState) -> AgentStateUpdate:
            if len(self.seen) == 0:
                self.seen.append(list(state.messages))
                await asyncio.sleep(2)
            return await super().respond(state)

    agent = _SlowModel([AIMessage(content="answer")])
    harness = LangGraphHarness(agent, MemorySaver())

    async def cancel_soon() -> bool:
        await asyncio.sleep(0.1)
        return await harness.cancel("c1")

    events, _ = await asyncio.gather(_collect(harness.run(_msg("go"), "c1", TURN)), cancel_soon())
    await _collect(harness.run(_msg("again"), "c1", TURN))

    assert events[-1] == TurnEnded(cancelled=True)
    *_, first, note, second = agent.seen[-1]
    assert (first.content, second.content) == ("go", "again")
    assert isinstance(note, HumanMessage)
    assert note.content == HARNESS_TURN_STOPPED_NOTE


async def test_a_finished_turn_gets_no_stop_note() -> None:
    harness, agent = _harness(AIMessage(content="done"), AIMessage(content="next"))
    await _collect(harness.run(_msg("go"), "c1", TURN))

    await harness._record_stop("c1")
    await _collect(harness.run(_msg("again"), "c1", TURN))

    assert [m.content for m in agent.seen[-1]] == ["go", "done", "again"]


async def test_model_error_ends_the_turn_with_its_kind() -> None:
    harness, _ = _harness(_status_error(openai.RateLimitError, 429))

    events = await _collect(harness.run(_msg("go"), "c1", TURN))

    assert events[-1].error == HarnessError(kind="rate_limited")


async def _until(condition) -> None:
    while not condition():  # noqa: ASYNC110 — polls state the code under test owns
        await asyncio.sleep(0.01)


async def _answering(stream: AsyncIterator, answers: dict[str, ApprovalDecision]) -> list:
    """Collect *stream*, answering each approval by its operation as the user would."""
    events = []
    async for event in stream:
        events.append(event)
        if isinstance(event, ApprovalRequested):
            await asyncio.sleep(0.01)
            assert pending_approvals.resolve(event.id, answers[event.request.operation])
    return events


async def test_approval_is_awaited_inside_the_tool() -> None:
    harness, agent = _harness(_call("guarded", {"command": "ls"}), AIMessage(content="done"))

    events = await _answering(
        harness.run(_msg("go"), "c1", TURN), {"ls": ApprovalDecision.APPROVED_ONCE}
    )

    [asked] = [e for e in events if isinstance(e, ApprovalRequested)]
    assert asked.request.operation == "ls"
    started = next(e for e in events if isinstance(e, ToolStarted))
    assert asked.parent_id == started.id  # nested under the tool that asked
    assert ApprovalResolved(asked.id, ApprovalDecision.APPROVED_ONCE, started.id) in events
    assert [type(e) for e in events].count(TurnStarted) == 1  # the turn never paused
    assert events[-1] == TurnEnded()
    assert agent.seen[1][-1].content == "ran ls"
    assert pending_approvals.for_context("c1") == []


async def test_a_denied_tool_ends_as_an_error() -> None:
    harness, agent = _harness(_call("guarded", {"command": "rm x"}), AIMessage(content="ok"))

    events = await _answering(
        harness.run(_msg("go"), "c1", TURN), {"rm x": ApprovalDecision.DENIED}
    )

    ended = next(e for e in events if isinstance(e, ToolEnded))
    assert ended.error is True
    assert "NOT executed" in agent.seen[1][-1].content


async def test_parallel_approvals_are_all_asked_at_once() -> None:
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "guarded", "args": {"command": "a"}, "id": "c1"},
            {"name": "guarded", "args": {"command": "b"}, "id": "c2"},
        ],
    )
    harness, agent = _harness(both, AIMessage(content="done"))

    async def answer_when_both_wait() -> None:
        await _until(lambda: len(pending_approvals.for_context("c1")) == 2)
        for p in pending_approvals.for_context("c1"):
            decision = (
                ApprovalDecision.DENIED
                if p.request.operation == "a"
                else (ApprovalDecision.APPROVED_ONCE)
            )
            assert pending_approvals.resolve(p.id, decision)

    events, _ = await asyncio.gather(
        _collect(harness.run(_msg("go"), "c1", TURN)), answer_when_both_wait()
    )

    errors = {e.id: e.error for e in events if isinstance(e, ToolEnded)}
    assert sorted(errors.values()) == [False, True]
    assert [type(e) for e in events].count(TurnStarted) == 1
    results = sorted(m.content for m in agent.seen[1][-2:])
    assert "NOT executed" in results[0]
    assert results[1] == "ran b"


async def test_a_second_answer_to_the_same_request_is_refused() -> None:
    harness, agent = _harness(
        _call("guarded", {"command": "one"}),
        _call("guarded", {"command": "two"}, call_id="c2"),
        AIMessage(content="done"),
    )
    answered: list[bool] = []

    async def double_tap() -> None:
        first_id = None
        while True:
            waiting = pending_approvals.for_context("c1")
            if waiting and first_id is None:
                first_id = waiting[0].id
                answered.append(pending_approvals.resolve(first_id, ApprovalDecision.DENIED))
                answered.append(pending_approvals.resolve(first_id, ApprovalDecision.APPROVED_ONCE))
            elif waiting and waiting[0].id != first_id:
                pending_approvals.resolve(waiting[0].id, ApprovalDecision.DENIED)
                return
            await asyncio.sleep(0.01)

    await asyncio.gather(_collect(harness.run(_msg("go"), "c1", TURN)), double_tap())

    assert answered == [True, False]
    # The second request was never approved by the stray tap.
    assert all("NOT executed" in m.content for m in agent.seen[2] if isinstance(m, ToolMessage))


async def test_an_unattended_turn_denies_approvals_at_once() -> None:
    harness, agent = _harness(_call("guarded", {"command": "ls"}), AIMessage(content="ok"))
    unattended = TurnContext(reply_to=TURN.reply_to, attended=False)

    events = await _collect(harness.run(_msg("go"), "c1", unattended))

    assert not any(isinstance(e, ApprovalRequested) for e in events)
    assert "nobody is present" in agent.seen[1][-1].content


async def test_an_unanswered_approval_times_out() -> None:
    harness, agent = _harness(_call("guarded", {"command": "ls"}), AIMessage(content="ok"))

    with patch("aug.core.harness.tool_context.APPROVAL_TIMEOUT_SECONDS", 0.05):
        events = await _collect(harness.run(_msg("go"), "c1", TURN))

    resolved = next(e for e in events if isinstance(e, ApprovalResolved))
    assert resolved.decision == ApprovalDecision.TIMED_OUT
    assert "in time" in agent.seen[1][-1].content
    assert pending_approvals.for_context("c1") == []


async def test_stop_while_waiting_for_approval_ends_the_turn() -> None:
    harness, _ = _harness(_call("guarded", {"command": "ls"}), AIMessage(content="never"))

    async def stop_once_asked() -> None:
        await _until(lambda: pending_approvals.for_context("c1"))
        await harness.cancel("c1")

    events, _ = await asyncio.gather(
        _collect(harness.run(_msg("go"), "c1", TURN)), stop_once_asked()
    )

    assert events[-1].cancelled is True
    assert pending_approvals.for_context("c1") == []  # a late tap finds nothing


def test_status_from_outside_a_tool_gets_no_parent() -> None:
    raw = {
        "event": "on_custom_event",
        "name": "aug_event",
        "run_id": "node-run",
        "data": {"event": StatusUpdate(text="🗜")},
    }
    progress = {**raw, "data": {"event": ToolProgress(text="step")}}

    assert parse_event(raw) == StatusUpdate(text="🗜")
    assert parse_event(progress) == ToolProgress(text="step", parent_id="node-run")


def test_a_file_sent_from_a_tool_belongs_to_that_tool(tmp_path) -> None:
    file = FileEmitted(
        file_id="0123456789abcdef", path=tmp_path / "x", media_type="text/plain", filename="x"
    )
    raw = {
        "event": "on_custom_event",
        "name": "aug_event",
        "run_id": "tool-run",
        "data": {"event": file},
    }

    assert parse_event(raw) == replace(file, parent_id="tool-run")


async def test_oneshot_calls_the_agent_model() -> None:
    harness, agent = _harness()
    agent.model_name = "some-model"
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="answer"))

    with patch("aug.core.harness.langgraph.build_chat_model", return_value=model) as build:
        assert await harness.oneshot("question") == "answer"

    build.assert_called_once_with("some-model", temperature=0.0)
    assert model.ainvoke.call_args.args[0] == [HumanMessage(content="question")]


async def test_oneshot_calls_the_model_of_the_tier_asked_for() -> None:
    agent = AugAgent(model="strong-model")
    harness = LangGraphHarness(agent, MemorySaver(), models={ModelTier.FAST: "fast-model"})
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="answer"))

    with patch("aug.core.harness.langgraph.build_chat_model", return_value=model) as build:
        await harness.oneshot("question", ModelTier.FAST, system="be brief")
        await harness.oneshot("question")  # STRONG, not mapped: the agent's own model

    assert [c.args[0] for c in build.call_args_list] == ["fast-model", "strong-model"]
    first = model.ainvoke.call_args_list[0].args[0]
    assert first == [SystemMessage(content="be brief"), HumanMessage(content="question")]


# ---------------------------------------------------------------------------
# One run lock for interfaces, approval resumes and pushes
# ---------------------------------------------------------------------------


async def _drain(_thread_id: str, stream: AsyncIterator) -> None:
    await _collect(stream)


async def test_push_waits_for_the_interface_turn_holding_the_run_lock() -> None:
    from aug.core.dispatch import fire_push

    iface = MagicMock()
    iface.resolve_context = AsyncMock(return_value="tg-1-0")
    iface.get_agent_version = MagicMock(return_value="v")
    iface.conversation_id = MagicMock(return_value="tg-1")
    iface.send_proactive_stream = AsyncMock(side_effect=_drain)
    app = MagicMock()
    app.state.interfaces = {"telegram": iface}

    lock = run_registry.run_lock("tg-1-0")
    await lock.acquire()  # an interface turn is in flight
    with patch("aug.core.dispatch.get_harness", return_value=ScriptedHarness()):
        push = asyncio.create_task(
            fire_push(
                app, interface="telegram", context_id="tg-1-0", message="m", push_type="agent"
            )
        )
        await asyncio.sleep(0.05)
        assert not push.done()
        iface.send_proactive_stream.assert_not_awaited()
        lock.release()
        await asyncio.wait_for(push, 1)

    iface.send_proactive_stream.assert_awaited_once()


async def test_push_turn_error_raises_so_fire_task_retries() -> None:
    from aug.core.dispatch import PushTurnFailed, fire_push

    iface = MagicMock()
    iface.resolve_context = AsyncMock(return_value="tg-1-0")
    iface.conversation_id = MagicMock(return_value="tg-1")
    iface.send_proactive_stream = AsyncMock(side_effect=_drain)
    app = MagicMock()
    app.state.interfaces = {"telegram": iface}
    failing = ScriptedHarness([TurnEnded(error=HarnessError(kind="unavailable"))])

    with (
        patch("aug.core.dispatch.get_harness", return_value=failing),
        pytest.raises(PushTurnFailed),
    ):
        await fire_push(
            app, interface="telegram", context_id="tg-1-0", message="m", push_type="agent"
        )


async def test_a_failed_push_sends_nothing_before_the_retry() -> None:
    """The retry delivers the whole turn; the failed attempt's partial text must not."""
    from aug.core.dispatch import PushTurnFailed, fire_push

    sent: list[str] = []

    async def send_at_end(_context_id, stream) -> None:
        text = ""
        async for event in stream:
            if isinstance(event, TextDelta):
                text += event.text
        sent.append(text)

    iface = MagicMock()
    iface.resolve_context = AsyncMock(return_value="tg-1-0")
    iface.conversation_id = MagicMock(return_value="tg-1")
    iface.send_proactive_stream = send_at_end
    app = MagicMock()
    app.state.interfaces = {"telegram": iface}
    failing = ScriptedHarness(
        [
            TextDelta(text="Let me check"),
            TurnEnded(error=HarnessError(kind="unavailable")),
        ]
    )

    with (
        patch("aug.core.dispatch.get_harness", return_value=failing),
        pytest.raises(PushTurnFailed),
    ):
        await fire_push(
            app, interface="telegram", context_id="tg-1-0", message="m", push_type="agent"
        )

    assert sent == []


async def test_a_push_that_hangs_is_stopped_and_not_retried() -> None:
    from aug.core.dispatch import fire_push

    harness, _ = _harness(_call("stubborn", {"seconds": 5}), AIMessage(content="never"))
    iface = MagicMock()
    iface.resolve_context = AsyncMock(return_value="tg-1-0")
    iface.conversation_id = MagicMock(return_value="tg-1")
    iface.send_proactive_stream = AsyncMock(side_effect=_drain)
    iface.send_proactive = AsyncMock()
    app = MagicMock()
    app.state.interfaces = {"telegram": iface}

    with (
        patch("aug.core.dispatch.get_harness", return_value=harness),
        patch("aug.core.dispatch.UNATTENDED_TIMEOUT_SECONDS", 0.2),
    ):
        await asyncio.wait_for(
            fire_push(
                app, interface="telegram", context_id="tg-1-0", message="m", push_type="agent"
            ),
            2,
        )

    iface.send_proactive.assert_awaited_once()
    assert "stopped" in iface.send_proactive.await_args.args[1]


async def test_compact_waits_for_the_turn_in_flight() -> None:
    from aug.api.interfaces.rest import RestApiInterface

    harness = ScriptedHarness()
    lock = run_registry.run_lock("t1")
    await lock.acquire()
    with patch("aug.api.interfaces.base.get_harness", return_value=harness):
        task = asyncio.create_task(RestApiInterface()._execute_compact("t1", "v"))
        await asyncio.sleep(0.05)
        assert harness.compacted == []
        lock.release()
        assert await asyncio.wait_for(task, 1) is True
    assert harness.compacted == ["t1"]


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


async def test_inject_after_leftovers_are_collected_is_refused_not_lost() -> None:
    """Once TurnEnded is out, a new message must get its own turn, not a dead queue."""
    harness, _ = _harness(AIMessage(content="final"))
    accepted = None
    async for event in harness.run(_msg("start"), "c1", TURN):
        if isinstance(event, TurnEnded):
            accepted = await harness.inject(_msg("just after"), "c1")
    assert accepted is False


async def test_unattended_turn_takes_no_injections() -> None:
    harness, _ = _harness(AIMessage(content="push reply"))
    unattended = TurnContext(reply_to=TURN.reply_to, attended=False)
    stream = harness.run(_msg("push"), "c1", unattended)
    await anext(stream)

    assert await harness.inject(_msg("user typing"), "c1") is False
    assert run_registry.accepting("c1") is False
    await _collect(stream)


def _reflexing_harness(display: str = "🪄 light on"):
    harness, agent = _harness(AIMessage(content="ok"))
    reflex = AsyncMock(return_value=ReflexOutput(inject="[reflex] done", display=display))
    agent.reflexes = [reflex]
    return harness, reflex


async def test_reflex_display_reaches_the_event_stream() -> None:
    harness, reflex = _reflexing_harness()

    events = await _collect(harness.run(_msg("turn on the light"), "c1", TURN))

    reflex.assert_awaited_once()
    assert StatusUpdate(text="🪄 light on") in events


async def test_follow_up_turns_fire_no_reflexes() -> None:
    harness, reflex = _reflexing_harness()
    follow_up = TurnContext(reply_to=TURN.reply_to, attended=True, follow_up=True)

    await _collect(harness.run(_msg("late message"), "c1", follow_up))

    reflex.assert_not_awaited()


async def test_close_turn_never_drops_a_run_it_did_not_open() -> None:
    harness, _ = _harness()
    harness._open_turn("c1", TURN)
    newer = AgentRun()
    run_registry.set("c1", newer)

    harness._close_turn("c1")

    assert run_registry.get("c1") is newer


async def test_stop_run_cancels_whatever_agent_the_turn_uses() -> None:
    """The settings fallback ("default") is not an agent; /stop must still work."""
    from aug.api.interfaces.rest import RestApiInterface

    idle, running, after = ScriptedHarness(running=False), ScriptedHarness(), ScriptedHarness()
    with patch("aug.api.interfaces.base.harnesses", return_value=[idle, running, after]):
        assert await RestApiInterface().stop_run("rest-x") is True
    assert running.cancelled == ["rest-x"]
    assert after.cancelled == []  # one harness owns the turn; the rest aren't asked


async def test_unattended_turns_are_capped_at_a_few_steps() -> None:
    """A push that loops on tools stops at its low cap with step_limit."""
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(30)]
    harness, agent = _harness(*loop)
    loop = agent.replies
    unattended = TurnContext(reply_to=TURN.reply_to, attended=False)

    events = await _collect(harness.run(_msg("push"), "c1", unattended))

    assert events[-1].error == HarnessError(kind="step_limit")
    assert len(loop) == 30 - UNATTENDED_MAX_TOOL_ROUNDS  # stopped asking the model


async def test_attended_turns_get_room_for_long_work() -> None:
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(15)]
    harness, _ = _harness(*loop, AIMessage(content="done"))

    events = await _collect(harness.run(_msg("chat"), "c1", TURN))

    assert events[-1] == TurnEnded()


async def test_a_chat_turn_looping_on_tools_is_stopped_at_the_cap() -> None:
    """A model that keeps calling tools (say, a note after every note) is stopped."""
    total = ATTENDED_MAX_TOOL_ROUNDS + 20
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(total)]
    harness, agent = _harness(*loop)
    loop = agent.replies

    events = await _collect(harness.run(_msg("chat"), "c1", TURN))

    assert events[-1].error == HarnessError(kind="step_limit")
    assert len(loop) == total - ATTENDED_MAX_TOOL_ROUNDS


async def test_a_message_sent_as_the_cap_fires_is_kept() -> None:
    n = ATTENDED_MAX_TOOL_ROUNDS
    loop = [_call("slow", {"seconds": 0.3 if i == n - 1 else 0}, f"c{i}") for i in range(n)]
    harness, agent = _harness(*loop)

    async def inject_during_the_last_round() -> bool:
        while len(agent.replies) > 0:  # noqa: ASYNC110 — the fake agent has no event
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return await harness.inject(_msg("what are you doing?"), "c1")

    events, injected = await asyncio.gather(
        _collect(harness.run(_msg("chat"), "c1", TURN)), inject_during_the_last_round()
    )

    assert injected is True
    assert events[-1].error == HarnessError(kind="step_limit")
    assert events[-1].unconsumed == [_msg("what are you doing?")]


async def test_the_next_turn_knows_the_last_one_hit_the_cap() -> None:
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(ATTENDED_MAX_TOOL_ROUNDS)]
    harness, agent = _harness(*loop, AIMessage(content="sorry"))
    await _collect(harness.run(_msg("chat"), "c1", TURN))

    events = await _collect(harness.run(_msg("again"), "c1", TURN))

    assert events[-1] == TurnEnded()
    assert [m.content for m in agent.seen[-1][-2:]] == [HARNESS_STEP_LIMIT_NOTE, "again"]


async def test_what_the_user_sent_as_the_cap_fired_gets_its_own_turn() -> None:
    iface = RestApiInterface()
    capped = TurnEnded(error=HarnessError(kind="step_limit"), unconsumed=[_msg("hey")])
    iface._deliver = AsyncMock(side_effect=[capped, TurnEnded()])
    incoming = IncomingMessage(
        parts=[TextContent(text="go")],
        interface="rest_api",
        sender_id="1",
        context_id="rest-c1",
        agent_version="x",
    )

    with patch("aug.api.interfaces.base._harness_for", return_value=ScriptedHarness()):
        await iface._converse(incoming, _msg("go"), MagicMock())

    assert iface._deliver.await_count == 2
    assert iface._deliver.await_args_list[1].args[1].text == "hey"


async def test_a_stopped_turn_tells_the_user_it_stopped_not_that_it_failed() -> None:
    from aug.api.interfaces.rest import RestApiInterface

    iface = RestApiInterface()

    async def render(stream, _context) -> None:
        await _collect(stream)

    iface.send_stream = render
    iface.send_message = AsyncMock()
    harness = ScriptedHarness([TurnEnded(cancelled=True)])

    await iface._deliver(harness, _msg("go"), "c1", TURN, MagicMock())

    iface.send_message.assert_awaited_once()
    assert iface.send_message.await_args.args[0] == HARNESS_STOPPED_MESSAGE
