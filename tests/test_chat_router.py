"""REST chat API: requests name the chat with context_id; there is no thread_id alias."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.rest import RestApiInterface
from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app
from tests.test_hooks import _async_ctx

_HEADERS = {"X-API-Key": "test-api-key"}


@pytest.fixture()
def client():
    with (
        patch("aug.app.create_pool", new=AsyncMock(return_value=MagicMock(close=AsyncMock()))),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(MagicMock())),
        patch("aug.app.init_workspace"),
        patch("aug.app.job_runner", new=MagicMock(start=AsyncMock(), stop=AsyncMock())),
        patch("aug.app.start_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.set_app"),
        patch("aug.app.set_pool"),
        patch("aug.app.serve_gateway", new=AsyncMock()),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
        patch.object(RestApiInterface, "invoke", new=AsyncMock(return_value=("hi back", []))),
        TestClient(create_app()) as c,
    ):
        yield c


def test_invoke_takes_context_id_and_echoes_it(client):
    response = client.post(
        "/chat/invoke",
        json={"context_id": "rest-1", "message": "hi", "agent": "fake"},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["context_id"] == "rest-1"
    assert response.json()["response"] == "hi back"


def test_invoke_rejects_the_old_thread_id_field(client):
    response = client.post(
        "/chat/invoke",
        json={"thread_id": "rest-1", "message": "hi", "agent": "fake"},
        headers=_HEADERS,
    )
    assert response.status_code == 422
