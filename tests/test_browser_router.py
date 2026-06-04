"""Tests for the /browser router and WebSocket auth.

Behaviors under test:
  - verify_ws_credential accepts a valid Mini App JWT, rejects the API key/garbage
  - GET /browser/status reports availability from the hub
  - WS /browser/stream rejects a bad token with code 4401
  - WS /browser/stream closes with 4404 when the browser view is unconfigured
  - WS /browser/stream streams frames to an authorised client
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aug.api.interfaces.telegram import TelegramInterface
from aug.api.security import verify_ws_credential
from aug.app import create_app
from aug.core.auth import create_jwt
from aug.core.browser_view import BrowserViewHub

_HEADERS = {"X-API-Key": "test-api-key"}
_SUBPROTOCOL = "aug.browser-view.v1"
_BOT = "bot-secret"


def _jwt() -> str:
    return create_jwt({"sub": "42", "src": "telegram"}, secret=_BOT)


def _settings_with_bot():
    return patch(
        "aug.api.security.get_settings",
        return_value=SimpleNamespace(TELEGRAM_BOT_TOKEN=_BOT),
    )


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
        patch("aug.app.stop_scheduler", new=AsyncMock()),
        patch("aug.app.set_push_app"),
        patch("aug.app.set_pool"),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c


class _FakeScreencast:
    def __init__(self, cdp_url, on_frame):
        self.on_frame = on_frame
        self.seed_value: bytes | None = b"FRAME"

    async def start(self):
        pass

    async def stop(self):
        pass

    async def seed(self):
        return self.seed_value


# ---------------------------------------------------------------------------
# verify_ws_credential
# ---------------------------------------------------------------------------


def test_verify_ws_credential_accepts_valid_jwt():
    with _settings_with_bot():
        assert verify_ws_credential(_jwt()) is True
        assert verify_ws_credential("bad.jwt.value") is False
        assert verify_ws_credential("") is False


def test_verify_ws_credential_rejects_api_key():
    # The permanent shared API key must NOT be accepted over the WebSocket.
    with _settings_with_bot():
        assert verify_ws_credential("test-api-key") is False


def test_verify_ws_credential_rejects_when_no_bot_token():
    with patch(
        "aug.api.security.get_settings",
        return_value=SimpleNamespace(TELEGRAM_BOT_TOKEN=None),
    ):
        assert verify_ws_credential(_jwt()) is False


# ---------------------------------------------------------------------------
# /browser/status
# ---------------------------------------------------------------------------


def test_status_unavailable_by_default(client):
    resp = client.get("/browser/status", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_status_requires_auth(client):
    assert client.get("/browser/status").status_code == 401


def test_status_available_when_configured(client):
    client.app.state.browser_view_hub = BrowserViewHub(
        "http://chromium:9222", screencast_factory=_FakeScreencast
    )
    resp = client.get("/browser/status", headers=_HEADERS)
    assert resp.json() == {"available": True}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_stream_rejects_bad_token(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/browser/stream", subprotocols=[_SUBPROTOCOL, "bad-token"]
        ) as ws:
            ws.receive_bytes()
    assert exc.value.code == 4401


def test_stream_rejects_api_key_as_token(client):
    # Even the valid API key must be rejected over the WebSocket.
    with _settings_with_bot(), pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/browser/stream", subprotocols=[_SUBPROTOCOL, "test-api-key"]
        ) as ws:
            ws.receive_bytes()
    assert exc.value.code == 4401


def test_stream_unavailable_when_unconfigured(client):
    with _settings_with_bot(), pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/browser/stream", subprotocols=[_SUBPROTOCOL, _jwt()]) as ws:
            ws.receive_bytes()
    assert exc.value.code == 4404


def test_stream_delivers_frames(client):
    client.app.state.browser_view_hub = BrowserViewHub(
        "http://chromium:9222", screencast_factory=_FakeScreencast
    )
    with _settings_with_bot():
        with client.websocket_connect("/browser/stream", subprotocols=[_SUBPROTOCOL, _jwt()]) as ws:
            assert ws.receive_bytes() == b"FRAME"
