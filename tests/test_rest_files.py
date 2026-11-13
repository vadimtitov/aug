"""Files over REST: upload, attach to a chat message, receive as an SSE event, download."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.base import FileContent, TextContent
from aug.api.interfaces.rest import RestApiInterface, _frame, _RestContext
from aug.api.interfaces.telegram import TelegramInterface
from aug.api.schemas.chat import Attachment, ChatRequest
from aug.app import create_app
from aug.core.harness import FileEmitted, TextDelta, TurnEnded
from aug.utils.storage import file_store
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
        TestClient(create_app()) as c,
    ):
        yield c


def test_upload_then_download_round_trip(client):
    uploaded = client.post(
        "/files/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=_HEADERS,
    )
    assert uploaded.status_code == 201
    body = uploaded.json()
    assert (body["filename"], body["mime_type"], body["size_bytes"]) == (
        "notes.txt",
        "text/plain",
        5,
    )

    meta = client.get(f"/files/{body['file_id']}", headers=_HEADERS)
    content = client.get(f"/files/{body['file_id']}/content", headers=_HEADERS)

    assert meta.json()["mime_type"] == "text/plain"
    assert content.content == b"hello"
    assert content.headers["content-type"].startswith("text/plain")
    assert "notes.txt" in content.headers["content-disposition"]


def test_an_upload_without_a_type_is_typed_by_its_name(client):
    uploaded = client.post(
        "/files/upload",
        files={"file": ("scan.pdf", b"%PDF", "application/octet-stream")},
        headers=_HEADERS,
    )
    assert uploaded.json()["mime_type"] == "application/pdf"


def test_an_upload_over_the_limit_is_refused(client):
    with patch("aug.api.routers.files.MAX_UPLOAD_BYTES", 4):
        response = client.post(
            "/files/upload", files={"file": ("big.bin", b"12345", "text/plain")}, headers=_HEADERS
        )

    assert response.status_code == 413


async def test_download_needs_the_api_key(client):
    stored = await file_store.save_outbound(b"secret", "a.txt", "text/plain")

    assert client.get(f"/files/{stored.id}/content").status_code == 401
    assert (
        client.get(f"/files/{stored.id}/content", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert client.get(f"/files/{stored.id}").status_code == 401


def test_unknown_file_is_404(client):
    assert client.get("/files/0123456789abcdef/content", headers=_HEADERS).status_code == 404
    assert client.get("/files/..%2Fsettings/content", headers=_HEADERS).status_code == 404


def test_chat_with_an_unknown_attachment_is_404(client):
    for endpoint in ("/chat/invoke", "/chat/stream"):
        response = client.post(
            endpoint,
            json={
                "context_id": "rest-1",
                "message": "look",
                "agent": "fake",
                "attachments": [{"file_id": "0123456789abcdef"}],
            },
            headers=_HEADERS,
        )
        assert response.status_code == 404, endpoint


def test_invoke_returns_the_files_the_agent_sent(client):
    files = [
        {
            "id": "0123456789abcdef",
            "filename": "chart.png",
            "media_type": "image/png",
            "caption": "",
            "url": "/files/0123456789abcdef/content",
        }
    ]
    with patch.object(RestApiInterface, "invoke", new=AsyncMock(return_value=("Here.", files))):
        response = client.post(
            "/chat/invoke",
            json={"context_id": "rest-1", "message": "chart please", "agent": "fake"},
            headers=_HEADERS,
        )

    assert response.json()["files"][0]["url"] == "/files/0123456789abcdef/content"


async def test_attachments_reach_the_agent_as_files():
    stored = await file_store.save_inbound(b"%PDF", "cv.pdf", "application/pdf")
    request = ChatRequest(
        context_id="rest-1", message="review this", attachments=[Attachment(file_id=stored.id)]
    )

    incoming = await RestApiInterface().receive_message(_RestContext(request, agent_version="x"))

    assert incoming.parts == [
        FileContent(path=str(stored.path), mime_type="application/pdf", filename="cv.pdf"),
        TextContent(text="review this"),
    ]


def test_a_file_event_has_its_own_sse_frame(tmp_path):
    event = FileEmitted(
        file_id="0123456789abcdef",
        path=tmp_path / "x.png",
        media_type="image/png",
        filename="chart.png",
        caption="Q3",
        parent_id="t1",
    )

    assert _frame(event) == (
        "file",
        {
            "id": "0123456789abcdef",
            "filename": "chart.png",
            "media_type": "image/png",
            "caption": "Q3",
            "url": "/files/0123456789abcdef/content",
            "parent_id": "t1",
        },
    )


async def test_invoke_collects_files_next_to_the_text(tmp_path):
    ctx = _RestContext(ChatRequest(context_id="rest-1", message="x"), agent_version="x")
    for event in (
        TextDelta(text="Here it is."),
        FileEmitted(
            file_id="0123456789abcdef",
            path=tmp_path / "x.png",
            media_type="image/png",
            filename="chart.png",
        ),
        TurnEnded(),
    ):
        if frame := _frame(event):
            ctx.put(*frame)
    ctx.close()

    text, files = await ctx.collect()

    assert text == "Here it is."
    assert [f.filename for f in files] == ["chart.png"]
