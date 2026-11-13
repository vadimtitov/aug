"""Answering approval requests: Telegram buttons, REST endpoints, and typed replies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from aug.api.interfaces.rest import RestApiInterface, _frame, _RestContext
from aug.api.routers.chat import router as chat_router
from aug.api.schemas.chat import ChatRequest
from aug.core.harness import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
    HarnessError,
    StatusUpdate,
    TextDelta,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnEnded,
    TurnStarted,
    pending_approvals,
)

_HEADERS = {"X-API-Key": "test-api-key"}
REQUEST = ApprovalRequest(tool_name="run_ssh", resource="box", operation="ls")


@pytest.fixture(autouse=True)
def clean_pending():
    pending_approvals.clear()
    yield
    pending_approvals.clear()


async def _open(context_id: str) -> tuple[str, asyncio.Task]:
    pending = pending_approvals.open(context_id, REQUEST)
    waiter = asyncio.create_task(pending_approvals.wait(pending.id))
    await asyncio.sleep(0)
    return pending.id, waiter


# ---------------------------------------------------------------------------
# Telegram buttons
# ---------------------------------------------------------------------------


def _press(data: str, chat_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = chat_id
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


@pytest.fixture()
def telegram():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    with patch("aug.api.interfaces.telegram.utils.is_allowed", return_value=True):
        yield TelegramInterface()


async def test_a_button_answers_its_own_request(telegram) -> None:
    approval_id, waiter = await _open("tg-42-0")

    await telegram._handle_approval_callback(_press(f"ap:{approval_id}:a"), MagicMock())

    assert await waiter == ApprovalDecision.APPROVED_ALWAYS


async def test_a_second_tap_is_already_handled(telegram) -> None:
    approval_id, waiter = await _open("tg-42-0")
    await telegram._handle_approval_callback(_press(f"ap:{approval_id}:d"), MagicMock())
    await waiter

    second = _press(f"ap:{approval_id}:o")
    await telegram._handle_approval_callback(second, MagicMock())

    second.callback_query.edit_message_text.assert_awaited_with(
        "⚠️ This approval has already been resolved."
    )


async def test_a_button_from_another_chat_is_refused(telegram) -> None:
    approval_id, waiter = await _open("tg-42-0")

    stranger = _press(f"ap:{approval_id}:o", chat_id=7)
    await telegram._handle_approval_callback(stranger, MagicMock())

    stranger.callback_query.edit_message_text.assert_awaited_with("❌ Unauthorized.")
    assert not waiter.done()
    waiter.cancel()


async def test_an_old_style_button_is_already_handled(telegram) -> None:
    old = _press("approval:tg-42-0:approved_once:v13")

    await telegram._handle_approval_callback(old, MagicMock())

    old.callback_query.edit_message_text.assert_awaited_with(
        "⚠️ This approval has already been resolved."
    )


# ---------------------------------------------------------------------------
# A typed reply instead of a tap
# ---------------------------------------------------------------------------


async def test_a_typed_message_answers_a_waiting_approval_and_joins_the_turn() -> None:
    iface = RestApiInterface()
    _, waiter = await _open("rest-1")
    harness = MagicMock(inject=AsyncMock(return_value=True))
    ctx = _RestContext(
        request=ChatRequest(context_id="rest-1", message="no, wait"), agent_version="x"
    )

    with patch("aug.api.interfaces.base._harness_for", return_value=harness):
        await iface.run(ctx)

    assert await waiter == ApprovalDecision.REPLIED
    harness.inject.assert_awaited_once()


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def test_every_harness_event_has_its_own_sse_frame() -> None:
    frames = [
        _frame(TurnStarted(context_id="c")),
        _frame(TextDelta(text="hi")),
        _frame(ToolStarted(id="t1", name="run_ssh", args={"command": "ls"})),
        _frame(ToolProgress(text="step", parent_id="t1")),
        _frame(ToolEnded(id="t1", name="run_ssh", output="ok", error=True)),
        _frame(StatusUpdate(text="🗜")),
        _frame(ApprovalRequested(id="a1", request=REQUEST, parent_id="t1")),
        _frame(ApprovalResolved(id="a1", decision=ApprovalDecision.DENIED, parent_id="t1")),
        _frame(TurnEnded(error=HarnessError(kind="rate_limited"))),
    ]

    assert [name for name, _ in frames] == [
        "turn_started",
        "text_delta",
        "tool_started",
        "tool_progress",
        "tool_ended",
        "status",
        "approval_requested",
        "approval_resolved",
        "turn_ended",
    ]
    assert frames[6][1] == {
        "id": "a1",
        "tool_name": "run_ssh",
        "resource": "box",
        "operation": "ls",
        "parent_id": "t1",
    }
    assert frames[7][1]["decision"] == "denied"
    assert frames[8][1] == {"error": "rate_limited", "cancelled": False}


def test_a_subagents_text_is_not_part_of_the_reply() -> None:
    assert _frame(TextDelta(text="inner", parent_id="t1")) is None


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(chat_router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_rest_approve_answers_the_pending_request() -> None:
    approval_id, waiter = await _open("rest-1")
    client = _client()

    listed = await client.get("/chat/rest-1/approvals", headers=_HEADERS)
    response = await client.post(
        "/chat/rest-1/approve",
        json={"id": approval_id, "decision": "approved_once"},
        headers=_HEADERS,
    )
    again = await client.post(
        "/chat/rest-1/approve",
        json={"id": approval_id, "decision": "denied"},
        headers=_HEADERS,
    )

    assert listed.json() == [
        {"id": approval_id, "tool_name": "run_ssh", "resource": "box", "operation": "ls"}
    ]
    assert response.status_code == 204
    assert await waiter == ApprovalDecision.APPROVED_ONCE
    assert again.status_code == 409


async def test_rest_approve_refuses_a_request_from_another_thread() -> None:
    approval_id, waiter = await _open("rest-1")

    response = await _client().post(
        "/chat/rest-2/approve",
        json={"id": approval_id, "decision": "approved_once"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    assert not waiter.done()
    waiter.cancel()
