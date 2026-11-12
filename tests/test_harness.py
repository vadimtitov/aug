"""The Harness contract, and LangGraphHarness on a real graph with a scripted model."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import psycopg
import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from aug.core.agents.base_agent import BaseAgent
from aug.core.events import ChatModelStreamEvent, StatusEvent, ToolStartEvent
from aug.core.harness import (
    Address,
    HarnessError,
    Message,
    Sender,
    TextPart,
    TurnContext,
    TurnEnded,
    TurnStarted,
    error_text,
)
from aug.core.harness.langgraph import (
    UNATTENDED_MAX_TOOL_ROUNDS,
    LangGraphHarness,
    _harness_error,
)
from aug.core.prompts import HARNESS_ERROR_MESSAGES, MID_RUN_INJECTION_PREFIX
from aug.core.reflexes import ReflexOutput
from aug.core.run import AGENT_RUN_CONFIG_KEY, AgentRun, run_registry
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.approval import ApprovalDecision, requires_approval
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
    yield
    run_registry.clear()


# ---------------------------------------------------------------------------
# run() contract — any harness
# ---------------------------------------------------------------------------


class _Crashing(ScriptedHarness):
    async def _run(self, message, context_id, turn):
        yield ChatModelStreamEvent(delta="partial")
        raise RuntimeError("boom")


async def test_run_starts_with_turn_started_and_ends_with_turn_ended() -> None:
    events = await _collect(
        ScriptedHarness([ChatModelStreamEvent(delta="hi")]).run(_msg("x"), "c1", TURN)
    )

    assert events[0] == TurnStarted(context_id="c1")
    assert events[1] == ChatModelStreamEvent(delta="hi")
    assert events[-1] == TurnEnded()


async def test_an_exception_inside_run_becomes_turn_ended_internal() -> None:
    events = await _collect(_Crashing().run(_msg("x"), "c1", TURN))

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].error == HarnessError(kind="internal")


async def test_the_harness_turn_ended_is_passed_through_once() -> None:
    ended = TurnEnded(error=HarnessError(kind="rate_limited"))
    harness = ScriptedHarness([ended, ChatModelStreamEvent(delta="never")])

    events = await _collect(harness.run(_msg("x"), "c1", TURN))

    assert [e for e in events if isinstance(e, TurnEnded)] == [ended]
    assert ChatModelStreamEvent(delta="never") not in events


async def test_turn_is_written_to_the_conversation_transcript(transcripts) -> None:
    harness = ScriptedHarness(
        [
            ToolStartEvent(run_id="r1", tool_name="note", args={"text": "x"}),
            ChatModelStreamEvent(delta="done"),
        ]
    )
    await _collect(harness.run(_msg("hello"), "tg-1-0", TURN))

    records = [json.loads(line) for line in (transcripts / "tg-1.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["turn_started", "tool_started", "turn_ended"]
    assert records[0]["text"] == "hello"
    assert records[1]["tool"] == "note"
    assert records[2]["text"] == "done"
    assert {r["context_id"] for r in records} == {"tg-1-0"}


async def test_a_failing_transcript_never_fails_the_turn(transcripts) -> None:
    transcripts.parent.mkdir(exist_ok=True)
    transcripts.write_text("a file where the directory should be")

    events = await _collect(
        ScriptedHarness([ChatModelStreamEvent(delta="hi")]).run(_msg("x"), "c1", TURN)
    )

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


@tool
@requires_approval(describe=lambda command: ("box", command))
async def guarded(command: str) -> str:
    """Run a command that needs approval."""
    return f"ran {command}"


class _ScriptAgent(BaseAgent):
    """Replies from a fixed list of AIMessages; records what each model call saw."""

    def __init__(self, replies: list[AIMessage | Exception]) -> None:
        super().__init__()
        self.tools = [slow, guarded]
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


async def test_model_error_ends_the_turn_with_its_kind() -> None:
    harness, _ = _harness(_status_error(openai.RateLimitError, 429))

    events = await _collect(harness.run(_msg("go"), "c1", TURN))

    assert events[-1].error == HarnessError(kind="rate_limited")


async def test_approval_pauses_then_resume_finishes_the_turn() -> None:
    harness, agent = _harness(_call("guarded", {"command": "ls"}), AIMessage(content="done"))

    paused = await _collect(harness.run(_msg("go"), "c1", TURN))
    request = await harness.pending_approval("c1")
    resumed = await _collect(harness.resume(ApprovalDecision.DENIED, "c1", TURN))

    assert paused[-1].approval == request
    assert request.operation == "ls"
    assert resumed[0] == TurnStarted(context_id="c1")
    assert resumed[-1] == TurnEnded()
    assert "NOT executed" in agent.seen[1][-1].content
    assert await harness.pending_approval("c1") is None


async def test_oneshot_calls_the_agent_model() -> None:
    harness, agent = _harness()
    agent.model_name = "some-model"
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="answer"))

    with patch("aug.core.harness.langgraph.build_chat_model", return_value=model) as build:
        assert await harness.oneshot("question") == "answer"

    build.assert_called_once_with("some-model", temperature=0.0)


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
    assert StatusEvent(text="🪄 light on") in events


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

    running = ScriptedHarness()
    with patch("aug.api.interfaces.base.harnesses", return_value=[ScriptedHarness(), running]):
        assert await RestApiInterface().stop_run("rest-x") is True
    assert running.cancelled == ["rest-x"]


async def test_unattended_turns_are_capped_at_a_few_steps() -> None:
    """A push that loops on tools stops at the cap with step_limit; a chat turn doesn't."""
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(30)]
    harness, agent = _harness(*loop)
    loop = agent.replies
    unattended = TurnContext(reply_to=TURN.reply_to, attended=False)

    events = await _collect(harness.run(_msg("push"), "c1", unattended))

    assert events[-1].error == HarnessError(kind="step_limit")
    assert len(loop) == 30 - UNATTENDED_MAX_TOOL_ROUNDS  # stopped asking the model


async def test_attended_turns_have_no_tool_round_cap() -> None:
    loop = [_call("slow", {"seconds": 0}, f"c{i}") for i in range(15)]
    harness, _ = _harness(*loop, AIMessage(content="done"))

    events = await _collect(harness.run(_msg("chat"), "c1", TURN))

    assert events[-1] == TurnEnded()
