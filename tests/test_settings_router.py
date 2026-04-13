"""Tests for GET/PUT /settings and GET /models endpoints."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------


def test_get_settings_returns_empty_when_no_file(client: TestClient) -> None:
    with patch("aug.api.routers.settings.get_all_settings", return_value={}):
        response = client.get("/settings", headers=_HEADERS)
    assert response.status_code == 200
    assert response.json() == {}


def test_get_settings_returns_current_settings(client: TestClient) -> None:
    data = {"tools": {"bash": {"blacklist": ["rm -rf"]}}}
    with patch("aug.api.routers.settings.get_all_settings", return_value=data):
        response = client.get("/settings", headers=_HEADERS)
    assert response.status_code == 200
    assert response.json() == data


def test_get_settings_requires_auth(client: TestClient) -> None:
    response = client.get("/settings")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# PUT /settings
# ---------------------------------------------------------------------------


def test_put_settings_writes_and_returns_data(client: TestClient) -> None:
    data = {"tools": {"bash": {"blacklist": ["rm -rf", "shutdown"]}}}
    with patch("aug.api.routers.settings.set_all_settings") as mock_write:
        response = client.put("/settings", json=data, headers=_HEADERS)
    assert response.status_code == 200
    assert response.json() == data
    mock_write.assert_called_once_with(data)


def test_put_settings_requires_auth(client: TestClient) -> None:
    response = client.put("/settings", json={})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


def test_get_models_returns_model_ids(client: TestClient) -> None:
    litellm_response = {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model"},
            {"id": "claude-sonnet-4-6", "object": "model"},
        ],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = litellm_response
    mock_response.raise_for_status = MagicMock()

    with patch("aug.api.routers.settings.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        response = client.get("/models", headers=_HEADERS)

    assert response.status_code == 200
    assert response.json() == ["gpt-4o", "claude-sonnet-4-6"]


def test_get_models_returns_error_when_litellm_unreachable(client: TestClient) -> None:
    with patch("aug.api.routers.settings.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        response = client.get("/models", headers=_HEADERS)

    assert response.status_code == 502
    assert "unavailable" in response.json()["detail"].lower()


def test_get_models_requires_auth(client: TestClient) -> None:
    response = client.get("/models")
    assert response.status_code == 401
