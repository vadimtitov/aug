"""Reflexes: run_reflexes, and Harness.run() running them before a turn."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from aug.core.harness import (
    Address,
    Message,
    Sender,
    StatusUpdate,
    TextDelta,
    TextPart,
    TurnContext,
    TurnEnded,
    TurnStarted,
)
from aug.core.reflexes import Exchange, Reflex, ReflexOutput, run_reflexes
from aug.core.transcript import Transcript
from tests.harness_fakes import ScriptedHarness

_TURN = TurnContext(reply_to=Address(interface="telegram", conversation_id="tg-1"), attended=True)


def _make_reflex(output: ReflexOutput | None) -> Reflex:
    return AsyncMock(return_value=output)


def _msg(text: str) -> Message:
    return Message(parts=[TextPart(text=text)], sender=Sender(interface="telegram", user_id="u"))


@pytest.fixture
def transcripts(tmp_path):
    with patch("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts"):
        yield tmp_path / "transcripts"


class _ReflexHarness(ScriptedHarness):
    def __init__(self, reflexes: list[Reflex], events=None) -> None:
        super().__init__(events or [TextDelta(text="ok")])
        self._reflexes = reflexes

    @property
    def reflexes(self):
        return self._reflexes


async def _collect(harness, message: Message, turn: TurnContext = _TURN) -> list:
    return [e async for e in harness.run(message, "tg-1-0", turn)]


# ---------------------------------------------------------------------------
# run_reflexes
# ---------------------------------------------------------------------------


async def test_run_reflexes_empty_list_returns_empty() -> None:
    assert await run_reflexes([], "hello", []) == []


async def test_run_reflexes_returns_non_none_outputs() -> None:
    reflexes = [
        _make_reflex(ReflexOutput(note="did thing A", display="A done")),
        _make_reflex(None),
        _make_reflex(ReflexOutput(note="did thing B")),
    ]
    results = await run_reflexes(reflexes, "query", [])
    assert results == [
        ReflexOutput(note="did thing A", display="A done"),
        ReflexOutput("did thing B"),
    ]


async def test_run_reflexes_passes_query_and_history() -> None:
    reflex = _make_reflex(None)
    history = [Exchange(role="user", text="previous message")]
    await run_reflexes([reflex], "turn on lights", history)
    reflex.assert_awaited_once_with("turn on lights", history)


async def test_run_reflexes_swallows_exception_from_one_reflex() -> None:
    async def bad_reflex(query, history) -> ReflexOutput | None:
        raise RuntimeError("boom")

    good = _make_reflex(ReflexOutput(note="ok"))
    assert await run_reflexes([bad_reflex, good], "query", []) == [ReflexOutput(note="ok")]


async def test_run_reflexes_swallows_timeout() -> None:
    async def slow_reflex(query, history) -> ReflexOutput | None:
        await asyncio.sleep(60)
        return ReflexOutput(note="never")

    good = _make_reflex(ReflexOutput(note="fast"))
    results = await run_reflexes([slow_reflex, good], "query", [], reflex_timeout=0.05)
    assert results == [ReflexOutput(note="fast")]


async def test_run_reflexes_runs_in_parallel() -> None:
    delay = 0.05

    async def reflex(query, history) -> ReflexOutput | None:
        await asyncio.sleep(delay)
        return ReflexOutput(note="done")

    t0 = time.monotonic()
    results = await run_reflexes([reflex, reflex, reflex], "query", [])
    assert len(results) == 3
    assert time.monotonic() - t0 < delay * 2


# ---------------------------------------------------------------------------
# Harness.run()
# ---------------------------------------------------------------------------


async def test_what_a_reflex_did_reaches_the_agent_and_the_user(transcripts) -> None:
    reflex = _make_reflex(ReflexOutput(note="light.turn_on on light.lamp", display="🪄 on"))
    harness = _ReflexHarness([reflex])

    events = await _collect(harness, _msg("turn on the lamp"))

    seen, _, _ = harness.turns[0]
    assert seen.parts[0] == TextPart(text="turn on the lamp")
    assert "light.turn_on on light.lamp" in seen.parts[1].text
    assert isinstance(events[0], TurnStarted)
    assert events[1] == StatusUpdate(text="🪄 on")
    assert isinstance(events[-1], TurnEnded)


async def test_the_transcript_keeps_the_users_own_words_and_records_the_reflex(
    transcripts,
) -> None:
    harness = _ReflexHarness([_make_reflex(ReflexOutput(note="did it"))])

    await _collect(harness, _msg("turn on the lamp"))

    records = [json.loads(line) for line in (transcripts / "tg-1.jsonl").read_text().splitlines()]
    started = next(r for r in records if r["event"] == "turn_started")
    assert started["text"] == "turn on the lamp"
    assert next(r for r in records if r["event"] == "reflexed")["notes"] == ["did it"]


async def test_a_reflex_with_nothing_to_do_leaves_the_message_alone(transcripts) -> None:
    harness = _ReflexHarness([_make_reflex(None)])
    message = _msg("what's the weather?")

    events = await _collect(harness, message)

    assert harness.turns[0][0] == message
    assert not any(isinstance(e, StatusUpdate) for e in events)


@pytest.mark.parametrize(
    "turn",
    [
        TurnContext(reply_to=_TURN.reply_to, attended=False),  # a push or scheduled task
        TurnContext(reply_to=_TURN.reply_to, attended=True, follow_up=True),
    ],
)
async def test_reflexes_only_run_when_someone_is_there(transcripts, turn) -> None:
    reflex = _make_reflex(ReflexOutput(note="did it"))
    harness = _ReflexHarness([reflex])

    await _collect(harness, _msg("turn on the lamp"), turn)

    reflex.assert_not_awaited()


async def test_a_reflex_sees_the_sessions_earlier_messages_without_the_current_one(
    transcripts,
) -> None:
    harness = _ReflexHarness([_make_reflex(None)], events=[TextDelta(text="Kitchen is off.")])
    await _collect(harness, _msg("turn off the kitchen light"))
    reflex = _make_reflex(None)
    harness._reflexes = [reflex]

    await _collect(harness, _msg("now turn it on"))

    reflex.assert_awaited_once_with(
        "now turn it on",
        [
            Exchange(role="user", text="turn off the kitchen light"),
            Exchange(role="assistant", text="Kitchen is off."),
        ],
    )


# ---------------------------------------------------------------------------
# Transcript.recent()
# ---------------------------------------------------------------------------


async def test_recent_returns_the_sessions_last_messages_oldest_first(tmp_path) -> None:
    t = Transcript("tg-1", root=tmp_path)
    for i in range(4):
        await t.append("turn_started", "s1", text=f"q{i}")
        await t.append("tool_started", "s1", tool="x")
        await t.append("turn_ended", "s1", text=f"a{i}")
    await t.append("turn_started", "s0", text="other session")
    await t.append("turn_ended", "s1", text="")  # a turn that said nothing

    records = await t.recent("s1", 3)

    assert [r["text"] for r in records] == ["a2", "q3", "a3"]


async def test_recent_leaves_out_turns_nobody_was_there_for(tmp_path) -> None:
    t = Transcript("tg-1", root=tmp_path)
    await t.append("turn_started", "s1", text="turn off the hall light", attended=True)
    await t.append("turn_ended", "s1", text="Done.")
    await t.append("turn_started", "s1", text="Scheduled task: check the weather", attended=False)
    await t.append("turn_ended", "s1", text="Synced.")

    records = await t.recent("s1", 4)

    assert [r["text"] for r in records] == ["turn off the hall light", "Done."]


async def test_recent_with_no_transcript_is_empty(tmp_path) -> None:
    assert await Transcript("tg-1", root=tmp_path).recent("s1", 3) == []


async def test_recent_reads_only_the_tail_of_a_long_transcript(tmp_path) -> None:
    t = Transcript("tg-1", root=tmp_path)
    with patch("aug.core.transcript._TAIL_BYTES", 200):
        for i in range(50):
            await t.append("turn_started", "s1", text=f"q{i}")
        records = await t.recent("s1", 100)
    assert records and len(records) < 50
    assert records[-1]["text"] == "q49"
