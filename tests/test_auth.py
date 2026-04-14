"""Tests for Telegram initData verification and JWT auth."""

import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app
from aug.core.auth import create_jwt, verify_telegram_init_data
from aug.utils.file_settings import AppSettings

_BOT_TOKEN = "123456:ABC-test-token"
_API_HEADERS = {"X-API-Key": "test-api-key"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_init_data(bot_token: str = _BOT_TOKEN, age_seconds: int = 0) -> str:
    """Build a correctly signed initData string."""
    auth_date = int(time.time()) - age_seconds
    user = json.dumps({"id": 1, "first_name": "Test", "username": "testuser"})
    params = {"auth_date": str(auth_date), "user": user}

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(params.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), digestmod=hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), digestmod=hashlib.sha256
    ).hexdigest()

    params["hash"] = computed_hash
    return urlencode(params)


# ---------------------------------------------------------------------------
# verify_telegram_init_data
# ---------------------------------------------------------------------------


def test_valid_init_data_returns_parsed_payload() -> None:
    init_data = _make_init_data()
    result = verify_telegram_init_data(init_data, _BOT_TOKEN)
    assert result["auth_date"] is not None


def test_tampered_hash_raises() -> None:
    init_data = _make_init_data() + "&hash=deadbeef"
    # hash appears twice — or just replace it
    init_data = _make_init_data().replace(_make_init_data().split("hash=")[1], "deadbeef")
    with pytest.raises(ValueError, match=r"[Ii]nvalid"):
        verify_telegram_init_data(init_data, _BOT_TOKEN)


def test_wrong_bot_token_raises() -> None:
    init_data = _make_init_data(bot_token=_BOT_TOKEN)
    with pytest.raises(ValueError, match=r"[Ii]nvalid"):
        verify_telegram_init_data(init_data, "wrong:token")


def test_expired_init_data_raises() -> None:
    init_data = _make_init_data(age_seconds=90000)  # 25 hours old
    with pytest.raises(ValueError, match=r"[Ee]xpir"):
        verify_telegram_init_data(init_data, _BOT_TOKEN)


def test_missing_hash_raises() -> None:
    params = {"auth_date": str(int(time.time())), "user": "{}"}
    init_data = urlencode(params)
    with pytest.raises(ValueError, match=r"[Hh]ash"):
        verify_telegram_init_data(init_data, _BOT_TOKEN)


# ---------------------------------------------------------------------------
# POST /auth/telegram
# ---------------------------------------------------------------------------


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
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c


def test_auth_telegram_valid_init_data_returns_token(client: TestClient) -> None:
    init_data = _make_init_data()
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        mock_settings.return_value.DEV_AUTH_BYPASS = False
        mock_settings.return_value.allowed_chat_ids = set()
        response = client.post("/auth/telegram", json={"init_data": init_data})
    assert response.status_code == 200
    assert "token" in response.json()


def test_auth_telegram_user_not_in_allowlist_returns_403(client: TestClient) -> None:
    init_data = _make_init_data()  # user id=1
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        mock_settings.return_value.DEV_AUTH_BYPASS = False
        mock_settings.return_value.allowed_chat_ids = {999}  # user 1 not in list
        response = client.post("/auth/telegram", json={"init_data": init_data})
    assert response.status_code == 403


def test_auth_telegram_user_in_allowlist_returns_token(client: TestClient) -> None:
    init_data = _make_init_data()  # user id=1
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        mock_settings.return_value.DEV_AUTH_BYPASS = False
        mock_settings.return_value.allowed_chat_ids = {1}
        response = client.post("/auth/telegram", json={"init_data": init_data})
    assert response.status_code == 200
    assert "token" in response.json()


def test_auth_telegram_invalid_init_data_returns_401(client: TestClient) -> None:
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        mock_settings.return_value.DEV_AUTH_BYPASS = False
        response = client.post("/auth/telegram", json={"init_data": "hash=invalid"})
    assert response.status_code == 401


def test_auth_telegram_dev_bypass_skips_verification_and_allowlist(client: TestClient) -> None:
    # Even with a restrictive allowlist, DEV_AUTH_BYPASS lets any user through.
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        mock_settings.return_value.DEV_AUTH_BYPASS = True
        mock_settings.return_value.allowed_chat_ids = {999}  # user 1 not in list — ignored
        response = client.post(
            "/auth/telegram",
            json={"init_data": "auth_date=1234567890&user={}&hash=dev_bypass"},
        )
    assert response.status_code == 200
    assert "token" in response.json()


def test_auth_telegram_no_bot_token_returns_503(client: TestClient) -> None:
    with patch("aug.api.routers.auth.get_settings") as mock_settings:
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = None
        response = client.post("/auth/telegram", json={"init_data": "anything"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# JWT Bearer auth on protected endpoints
# ---------------------------------------------------------------------------


def test_settings_accessible_with_valid_jwt(client: TestClient) -> None:
    token = create_jwt({"sub": "1"}, secret=_BOT_TOKEN)
    with (
        patch("aug.api.security.get_settings") as mock_settings,
        patch("aug.api.routers.settings.load_settings", return_value=AppSettings()),
    ):
        mock_settings.return_value.API_KEY = "test-api-key"
        mock_settings.return_value.TELEGRAM_BOT_TOKEN = _BOT_TOKEN
        response = client.get("/settings", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_settings_rejects_invalid_jwt(client: TestClient) -> None:
    response = client.get("/settings", headers={"Authorization": "Bearer not-a-valid-token"})
    assert response.status_code == 401


def test_settings_still_accepts_api_key(client: TestClient) -> None:
    with patch("aug.api.routers.settings.load_settings", return_value=AppSettings()):
        response = client.get("/settings", headers=_API_HEADERS)
    assert response.status_code == 200
