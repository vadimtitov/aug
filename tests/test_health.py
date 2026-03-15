"""Smoke test — proves pytest wiring works and the health endpoint responds."""

# conftest.py seeds required env vars before aug modules are imported.
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app


@asynccontextmanager
async def _async_ctx(value):
    yield value


@pytest.fixture()
def client():
    """Return a TestClient with all external dependencies mocked out."""
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()

    mock_checkpointer = MagicMock()

    with (
        patch("aug.app.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(mock_checkpointer)),
        patch("aug.app.init_memory_files"),
        patch("aug.app.start_consolidation_scheduler", new=AsyncMock()),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
