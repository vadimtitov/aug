"""Session status above the harness: context warnings, Telegram's /status and /compact,
and REST's status endpoint and turn_ended frame."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from aug.api.interfaces.base import BaseInterface, IncomingMessage
from aug.api.interfaces.rest import RestApiInterface, _frame
from aug.api.interfaces.telegram.status import _StatusMixin, render_status
from aug.api.routers.chat import router as chat_router
from aug.core.harness import (
    Address,
    ContextUsage,
    HarnessEvent,
    LimitUsage,
    Message,
    Sender,
    SessionStatus,
    TextPart,
    TurnContext,
    TurnEnded,
)
from aug.core.run import RunRegistry
from tests.harness_fakes import ScriptedHarness

_HEADERS = {"X-API-Key": "test-api-key"}
_TURN = TurnContext(reply_to=Address(interface="test", conversation_id="c"), attended=True)
_MESSAGE = Message(parts=[TextPart(text="hi")], sender=Sender(interface="test", user_id="u"))


# ---------------------------------------------------------------------------
# Context warnings
# ---------------------------------------------------------------------------


class _Interface(BaseInterface[list[str]]):
    """Collects what it would send into the list it is given as context."""

    async def receive_message(self, context: list[str]) -> IncomingMessage | None:
        return None

    async def send_stream(self, stream: AsyncIterator[HarnessEvent], context: list[str]) -> None:
        async for _ in stream:
            pass

    async def send_message(self, message: str, context: list[str]) -> None:
        context.append(message)

    async def send_notification(self, target_id: str, text: str) -> None:
        pass

    async def resolve_context(self, context_id, *, topic_name=None, chat_id=None) -> str:
        return context_id

    def conversation_id(self, context_id: str) -> str:
        return context_id

    async def send_proactive(self, context_id: str, text: str) -> None:
        pass

    async def announcement_threads(self) -> list[str]:
        return []

    async def send_proactive_stream(self, context_id, stream) -> None:
        pass


async def _turns(*contexts: ContextUsage | None) -> list[list[str]]:
    """What the user is sent after each of a run of turns ending with *contexts*."""
    interface = _Interface()
    sent = []
    with patch("aug.api.interfaces.base.run_registry", RunRegistry()):
        for context in contexts:
            harness = ScriptedHarness([TurnEnded(context=context)])
            out: list[str] = []
            await interface._deliver(harness, _MESSAGE, "c1", _TURN, out)
            sent.append(out)
    return sent


def _at(percent: int, compacts_at: int | None = None) -> ContextUsage:
    return ContextUsage(used=percent * 1000, window=100_000, compacts_at=compacts_at)


async def test_each_warning_level_is_announced_once_on_the_way_up() -> None:
    sent = await _turns(_at(50), _at(71), _at(75), _at(91), _at(95))
    assert sent[0] == [] and sent[2] == [] and sent[4] == []
    assert sent[1] == ["⚠️ Context is 71% full. /compact or /clear frees it."]
    assert sent[3] == ["⚠️ Context is 91% full. /compact or /clear frees it."]


async def test_after_a_compaction_a_level_is_announced_again() -> None:
    sent = await _turns(_at(75), _at(20), _at(75))
    assert len(sent[0]) == 1 and sent[1] == [] and len(sent[2]) == 1


async def test_levels_count_towards_where_the_session_compacts_itself() -> None:
    # Compacts at 80K: 70% of the way there is 56K.
    (sent,) = await _turns(_at(57, compacts_at=80_000))
    assert sent == [
        "⚠️ Context is 57% full. /compact or /clear frees it. It compacts itself at 80%."
    ]


async def test_a_turn_without_a_context_sends_no_warning() -> None:
    assert await _turns(None) == [[]]


def test_rest_never_warns_in_the_reply() -> None:
    # Its clients read the context from turn_ended; /chat/invoke's reply stays clean.
    assert RestApiInterface._warns_context is False and _Interface._warns_context is True


class _Stalling(ScriptedHarness):
    async def session_status(self, context_id: str) -> SessionStatus:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_a_stalling_status_read_does_not_hold_the_turn_end() -> None:
    harness = _Stalling()
    with patch("aug.core.harness.base.CONTEXT_AFTER_TURN_SECONDS", 0.05):
        events = [e async for e in harness.run(_MESSAGE, "c1", _TURN)]
    assert isinstance(events[-1], TurnEnded) and events[-1].context is None


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def test_status_shows_model_context_breakdown_limits_and_background_work() -> None:
    status = SessionStatus(
        model="claude-opus-5-5",
        context=ContextUsage(
            used=82_000,
            window=200_000,
            compacts_at=180_000,
            breakdown=(("messages", 61_000), ("mcp tools", 14_000), ("system prompt", 7_000)),
        ),
        limits=(
            LimitUsage(window="five_hour", utilization=0.22),
            LimitUsage(
                window="seven_day",
                utilization=0.77,
                resets_at=datetime(2026, 11, 21, 14, 0, tzinfo=UTC),
            ),
        ),
        background_tasks=2,
    )
    assert render_status("v14_claude_code", status) == "\n".join(
        [
            "<b>v14_claude_code</b> · <code>claude-opus-5-5</code>",
            "",
            "<code>▓▓▓▓░░░░░░</code> 41% · 82K / 200K",
            "Compacts itself at 90%",
            "<i>messages 61K · mcp tools 14K · system prompt 7K</i>",
            "",
            "5-hour limit: 22%",
            "Weekly limit: 77% · resets Sat 14:00 UTC",
            "",
            "⏳ 2 background tasks running",
        ]
    )


def test_an_estimated_context_says_so() -> None:
    status = SessionStatus(
        model="kimi", context=ContextUsage(used=500, window=1000, estimated=True)
    )
    assert "<code>▓▓▓▓▓░░░░░</code> ~50% · ~500 / 1K" in render_status("v12", status)


def test_a_session_not_started_yet_has_no_context() -> None:
    text = render_status("v12", SessionStatus(model="kimi"))
    assert "No context yet" in text


class _Chat(_StatusMixin):
    """Just enough of TelegramInterface for the status handlers' logic."""

    def __init__(self, harness: ScriptedHarness, after: ContextUsage | None) -> None:
        self.harness = harness
        self.after = after

    def get_agent_version(self, context_id: str) -> str:
        return "v12"

    async def session_status(self, context_id: str) -> SessionStatus | None:
        return await self.harness.session_status(context_id)

    async def _execute_compact(self, context_id: str, agent_version: str) -> bool:
        self.harness.status = SessionStatus(model="m", context=self.after)
        return True


@pytest.fixture
def agents():
    with patch("aug.api.interfaces.telegram.status.list_agents", return_value=["v12"]):
        yield


async def test_compact_reports_the_context_before_and_after(agents) -> None:
    harness = ScriptedHarness(status=SessionStatus(model="m", context=_at(80)))
    chat = _Chat(harness, after=ContextUsage(used=12_000, window=100_000))
    assert await chat._compact("c1") == "✅ Compacted: 80K → 12K tokens."


async def test_compact_without_a_context_to_compare_just_says_done(agents) -> None:
    chat = _Chat(ScriptedHarness(status=SessionStatus(model="m")), after=None)
    assert await chat._compact("c1") == "✅ Done."


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def test_turn_ended_frame_carries_the_context() -> None:
    usage = ContextUsage(used=500, window=1000, breakdown=(("messages", 500),), estimated=True)
    name, payload = _frame(TurnEnded(context=usage))
    assert name == "turn_ended"
    assert payload["context"] == {
        "used": 500,
        "window": 1000,
        "percent": 50,
        "compacts_at": None,
        "breakdown": {"messages": 500},
        "estimated": True,
    }


async def test_rest_status_endpoint() -> None:
    status = SessionStatus(
        model="kimi",
        context=ContextUsage(used=500, window=1000),
        limits=(LimitUsage(window="five_hour", utilization=0.5),),
    )
    harness = ScriptedHarness(status=status)
    app = FastAPI()
    app.include_router(chat_router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    with (
        patch("aug.api.routers.chat.list_agents", return_value=["v12"]),
        patch("aug.api.routers.chat.get_harness", return_value=harness),
    ):
        response = await client.get("/chat/rest-1/status?agent=v12", headers=_HEADERS)
        unknown = await client.get("/chat/rest-1/status?agent=nope", headers=_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "v12" and body["model"] == "kimi"
    assert body["context"]["percent"] == 50
    assert body["limits"] == [{"window": "five_hour", "utilization": 0.5, "resets_at": None}]
    assert unknown.status_code == 400
