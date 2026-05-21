"""Tests for POST /hooks/push.

Behaviors under test:
  - Missing / wrong API key → 401
  - type="forward", valid interface → 200, fire_push called with correct args
  - type="agent", valid interface → 202 (fire-and-forget)
  - Unknown interface name → 400
  - Missing required field (message) → 422
  - _guarded_push logs exceptions rather than dropping them silently
"""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app

_HEADERS = {"X-API-Key": "test-api-key"}


@asynccontextmanager
async def _async_ctx(value):
    yield value


@pytest.fixture()
def client():
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    mock_checkpointer = MagicMock()

    with (
        patch("aug.app.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(mock_checkpointer)),
        patch("aug.app.init_memory_files"),
        patch("aug.app.start_consolidation_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.start_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.set_push_app"),
        patch("aug.app.set_pool"),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
        patch("aug.api.routers.hooks.fire_push", new=AsyncMock()) as mock_fire,
    ):
        test_app = create_app()
        # Register a mock interface so "telegram" is resolvable
        mock_iface = AsyncMock()
        mock_iface.resolve_thread = AsyncMock(return_value="tg-123-0")
        mock_iface.send_proactive = AsyncMock()

        with TestClient(test_app, raise_server_exceptions=True) as c:
            test_app.state.interfaces["telegram"] = mock_iface
            yield c, mock_fire, mock_iface


# ── auth ──────────────────────────────────────────────────────────────────────


def test_push_no_api_key_returns_401(client):
    c, _, _ = client
    response = c.post("/hooks/push", json={"interface": "telegram", "message": "hi"})
    assert response.status_code == 401


def test_push_wrong_api_key_returns_401(client):
    c, _, _ = client
    response = c.post(
        "/hooks/push",
        json={"interface": "telegram", "message": "hi"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


# ── forward ───────────────────────────────────────────────────────────────────


def test_push_forward_returns_200(client):
    c, _mock_fire, _ = client
    response = c.post(
        "/hooks/push",
        json={"interface": "telegram", "message": "door opened", "type": "forward"},
        headers=_HEADERS,
    )
    assert response.status_code == 200


def test_push_forward_calls_fire_push_with_correct_args(client):
    c, mock_fire, _ = client
    c.post(
        "/hooks/push",
        json={"interface": "telegram", "message": "door opened", "type": "forward"},
        headers=_HEADERS,
    )
    mock_fire.assert_called_once()
    call_kwargs = mock_fire.call_args.kwargs
    assert call_kwargs["interface"] == "telegram"
    assert call_kwargs["message"] == "door opened"
    assert call_kwargs["push_type"] == "forward"


# ── agent ─────────────────────────────────────────────────────────────────────


def test_push_agent_returns_202(client):
    c, _mock_fire, _ = client
    response = c.post(
        "/hooks/push",
        json={"interface": "telegram", "message": "check camera", "type": "agent"},
        headers=_HEADERS,
    )
    assert response.status_code == 202


# ── unknown interface ─────────────────────────────────────────────────────────


def test_push_invalid_interface_name_returns_422(client):
    """Interface names outside the Literal are rejected by Pydantic before reaching the endpoint."""
    c, _, _ = client
    response = c.post(
        "/hooks/push",
        json={"interface": "whatsapp", "message": "hi"},
        headers=_HEADERS,
    )
    assert response.status_code == 422


# ── validation ────────────────────────────────────────────────────────────────


def test_push_missing_message_returns_422(client):
    c, _, _ = client
    response = c.post(
        "/hooks/push",
        json={"interface": "telegram"},
        headers=_HEADERS,
    )
    assert response.status_code == 422


# ── guarded push ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guarded_push_logs_exception_on_failure(caplog):
    """_guarded_push logs errors so fire-and-forget failures are never silent."""
    from aug.api.routers.hooks import PushRequest, _guarded_push

    payload = PushRequest(interface="telegram", message="test", type="agent")

    with (
        patch(
            "aug.api.routers.hooks.fire_push",
            AsyncMock(side_effect=RuntimeError("delivery exploded")),
        ),
        caplog.at_level(logging.ERROR, logger="aug.api.routers.hooks"),
    ):
        await _guarded_push(MagicMock(), payload)

    assert "push-agent failed" in caplog.text
