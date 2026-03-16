"""Unit tests for Gmail tools."""

from unittest.mock import MagicMock, patch

import pytest

from aug.core.tools.gmail import (
    _decode_body,
    _format_message,
    gmail_draft,
    gmail_read_thread,
    gmail_search,
    gmail_send,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(messages=None, thread=None, send_result=None, draft_result=None):
    """Build a mock Gmail service object."""
    svc = MagicMock()

    # messages().list().execute()
    svc.users().messages().list().execute.return_value = {"messages": messages or []}

    # messages().get().execute()
    svc.users().messages().get().execute.return_value = {
        "id": "msg1",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"},
            ]
        },
        "snippet": "snippet text",
    }

    # threads().get().execute()
    svc.users().threads().get().execute.return_value = thread or {"messages": []}

    # messages().send().execute()
    svc.users().messages().send().execute.return_value = send_result or {"id": "sent1"}

    # drafts().create().execute()
    svc.users().drafts().create().execute.return_value = draft_result or {"id": "draft1"}

    return svc


def _patch_service(svc):
    return patch("aug.core.tools.gmail._service", return_value=svc)


def _patch_no_token(account="primary"):
    return patch(
        "aug.core.tools.gmail.load_token",
        return_value=None,
    )


# ---------------------------------------------------------------------------
# _decode_body
# ---------------------------------------------------------------------------


def test_decode_body_plain_text():
    import base64

    data = base64.urlsafe_b64encode(b"Hello world").decode()
    payload = {"mimeType": "text/plain", "body": {"data": data}}
    assert _decode_body(payload) == "Hello world"


def test_decode_body_multipart():
    import base64

    data = base64.urlsafe_b64encode(b"Nested body").decode()
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": data}},
        ],
    }
    assert _decode_body(payload) == "Nested body"


def test_decode_body_empty():
    assert _decode_body({"mimeType": "text/html", "body": {}}) == ""


# ---------------------------------------------------------------------------
# _format_message
# ---------------------------------------------------------------------------


def test_format_message_includes_fields():
    import base64

    data = base64.urlsafe_b64encode(b"Email body content").decode()
    msg = {
        "snippet": "fallback snippet",
        "payload": {
            "mimeType": "text/plain",
            "body": {"data": data},
            "headers": [
                {"name": "From", "value": "alice@example.com"},
                {"name": "Subject", "value": "Hello"},
                {"name": "Date", "value": "2024-01-01"},
            ],
        },
    }
    result = _format_message(msg)
    assert "alice@example.com" in result
    assert "Hello" in result
    assert "Email body content" in result


def test_format_message_falls_back_to_snippet():
    msg = {
        "snippet": "just a snippet",
        "payload": {
            "mimeType": "text/html",
            "body": {},
            "headers": [],
        },
    }
    result = _format_message(msg)
    assert "just a snippet" in result


# ---------------------------------------------------------------------------
# gmail_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gmail_search_returns_results():
    svc = _make_service(messages=[{"id": "msg1"}])
    with _patch_service(svc):
        result = await gmail_search.ainvoke({"query": "is:unread"})
    assert "msg1" in result
    assert "Found 1 message(s)" in result


@pytest.mark.asyncio
async def test_gmail_search_no_results():
    svc = _make_service(messages=[])
    with _patch_service(svc):
        result = await gmail_search.ainvoke({"query": "is:unread"})
    assert "No messages found" in result


@pytest.mark.asyncio
async def test_gmail_search_no_token():
    with _patch_no_token():
        with patch("aug.core.tools.gmail.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "http://localhost:8012"
            result = await gmail_search.ainvoke({"query": "test"})
    assert "not connected" in result
    assert "http://localhost:8012/auth/gmail" in result


@pytest.mark.asyncio
async def test_gmail_search_api_error():
    svc = MagicMock()
    svc.users().messages().list().execute.side_effect = Exception("API error")
    with _patch_service(svc):
        result = await gmail_search.ainvoke({"query": "test"})
    assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# gmail_read_thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gmail_read_thread_returns_content():
    thread = {
        "messages": [
            {
                "snippet": "",
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": ""},
                    "headers": [
                        {"name": "From", "value": "bob@example.com"},
                        {"name": "Subject", "value": "Re: Hello"},
                        {"name": "Date", "value": "2024-01-02"},
                    ],
                },
            }
        ]
    }
    svc = _make_service(thread=thread)
    with _patch_service(svc):
        result = await gmail_read_thread.ainvoke({"thread_id": "thread123"})
    assert "thread123" in result
    assert "bob@example.com" in result


@pytest.mark.asyncio
async def test_gmail_read_thread_empty():
    svc = _make_service(thread={"messages": []})
    with _patch_service(svc):
        result = await gmail_read_thread.ainvoke({"thread_id": "empty_thread"})
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_gmail_read_thread_no_token():
    with _patch_no_token():
        with patch("aug.core.tools.gmail.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "http://localhost:8012"
            result = await gmail_read_thread.ainvoke({"thread_id": "t1"})
    assert "not connected" in result


# ---------------------------------------------------------------------------
# gmail_send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gmail_send_success():
    svc = _make_service(send_result={"id": "sent123"})
    with _patch_service(svc):
        result = await gmail_send.ainvoke(
            {
                "to": "bob@example.com",
                "subject": "Hi",
                "body": "Hello Bob",
            }
        )
    assert "bob@example.com" in result
    assert "Hi" in result


@pytest.mark.asyncio
async def test_gmail_send_no_token():
    with _patch_no_token():
        with patch("aug.core.tools.gmail.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "http://localhost:8012"
            result = await gmail_send.ainvoke(
                {
                    "to": "bob@example.com",
                    "subject": "Hi",
                    "body": "Hello",
                }
            )
    assert "not connected" in result


@pytest.mark.asyncio
async def test_gmail_send_api_error():
    svc = MagicMock()
    svc.users().messages().send().execute.side_effect = Exception("SMTP error")
    with _patch_service(svc):
        result = await gmail_send.ainvoke(
            {
                "to": "bob@example.com",
                "subject": "Hi",
                "body": "Hello",
            }
        )
    assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# gmail_draft
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gmail_draft_success():
    svc = _make_service(draft_result={"id": "draft42"})
    with _patch_service(svc):
        result = await gmail_draft.ainvoke(
            {
                "to": "alice@example.com",
                "subject": "Draft subject",
                "body": "Draft body",
            }
        )
    assert "draft42" in result
    assert "alice@example.com" in result


@pytest.mark.asyncio
async def test_gmail_draft_no_token():
    with _patch_no_token():
        with patch("aug.core.tools.gmail.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "http://localhost:8012"
            result = await gmail_draft.ainvoke(
                {
                    "to": "alice@example.com",
                    "subject": "Draft",
                    "body": "Body",
                }
            )
    assert "not connected" in result


@pytest.mark.asyncio
async def test_gmail_draft_api_error():
    svc = MagicMock()
    svc.users().drafts().create().execute.side_effect = Exception("API down")
    with _patch_service(svc):
        result = await gmail_draft.ainvoke(
            {
                "to": "alice@example.com",
                "subject": "Draft",
                "body": "Body",
            }
        )
    assert "failed" in result.lower()
