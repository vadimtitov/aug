"""Unit tests for Gmail tools."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from aug.core.tools.gmail import (
    _PREVIEW_CHARS,
    _THREAD_BODY_CHARS,
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


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def _message(payload: dict, snippet: str = "") -> dict:
    headers = [
        {"name": "From", "value": "news@example.com"},
        {"name": "Subject", "value": "Newsletter"},
        {"name": "Date", "value": "2024-01-03"},
    ]
    return {"snippet": snippet, "payload": {**payload, "headers": headers}}


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


def test_decode_body_html_only_is_converted_to_markdown():
    html = (
        "<html><head><title>Ignored</title><style>p {color: red}</style></head>"
        "<body><h1>Big sale</h1><p>Everything&nbsp;is <b>half</b> price.</p>"
        '<img src="https://t.example/pixel.gif"><a href="https://shop.example">Shop now</a>'
        "</body></html>"
    )
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [{"mimeType": "text/html", "body": {"data": _b64(html)}}],
    }
    body = _decode_body(payload)
    assert "# Big sale" in body
    assert "Everything is **half** price." in body
    assert "[Shop now](https://shop.example)" in body
    assert "Ignored" not in body
    assert "color" not in body
    assert "pixel.gif" not in body


def test_decode_body_prefers_plain_over_html():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": _b64("<p>html version</p>")}},
            {"mimeType": "text/plain", "body": {"data": _b64("plain version")}},
        ],
    }
    assert _decode_body(payload) == "plain version"


def test_decode_body_cleans_whitespace_and_placeholders():
    raw = (
        "Hello\u00a0\u00a0 \tthere\u200b\u200c friend\r\n"
        "[Image]\n\n\n\n\n"
        "  [Header]  \n"
        "Second\t\tparagraph   \n\n\n\nThird [Image] inline stays"
    )
    body = _decode_body({"mimeType": "text/plain", "body": {"data": _b64(raw)}})
    assert body == "Hello there friend\n\nSecond paragraph\n\nThird [Image] inline stays"


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


def test_format_message_preview_stays_short():
    words = "word " * 1000
    result = _format_message(_message({"mimeType": "text/plain", "body": {"data": _b64(words)}}))
    body = result.split("\n\n", 1)[1]
    preview, marker = body.rsplit("\n", 1)
    assert len(preview) <= _PREVIEW_CHARS
    assert preview.endswith("word")
    assert marker == f"[truncated, {len(words.strip()) - len(preview)} more chars]"


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
async def test_gmail_search_lists_headers_only():
    svc = _make_service(messages=[{"id": "msg1"}])
    with _patch_service(svc):
        result = await gmail_search.ainvoke({"query": "is:unread"})
    assert len(result) < _PREVIEW_CHARS
    assert svc.users().messages().get.call_args.kwargs["format"] == "metadata"


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
    assert "GMAIL AUTH FAILED" in result
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
async def test_gmail_read_thread_keeps_long_plain_body():
    body = " ".join(f"word{i}" for i in range(1000))  # ~6.9k chars
    thread = {"messages": [_message({"mimeType": "text/plain", "body": {"data": _b64(body)}})]}
    with _patch_service(_make_service(thread=thread)):
        result = await gmail_read_thread.ainvoke({"thread_id": "t1"})
    assert body in result
    assert "truncated" not in result


@pytest.mark.asyncio
async def test_gmail_read_thread_html_only_body():
    html = "<p>" + "Only HTML here. " * 50 + "</p><p>The end.</p>"
    thread = {
        "messages": [
            _message({"mimeType": "text/html", "body": {"data": _b64(html)}}, snippet="snip")
        ]
    }
    with _patch_service(_make_service(thread=thread)):
        result = await gmail_read_thread.ainvoke({"thread_id": "t1"})
    assert "The end." in result
    assert "snip" not in result


@pytest.mark.asyncio
async def test_gmail_read_thread_marks_truncation_of_huge_body():
    body = "word " * 10_000  # 50k chars
    thread = {"messages": [_message({"mimeType": "text/plain", "body": {"data": _b64(body)}})]}
    with _patch_service(_make_service(thread=thread)):
        result = await gmail_read_thread.ainvoke({"thread_id": "t1"})
    text = result.split("Subject: Newsletter\n\n", 1)[1]
    kept, marker = text.rsplit("\n", 1)
    assert _PREVIEW_CHARS < len(kept) <= _THREAD_BODY_CHARS
    assert kept.endswith("word")
    assert marker == f"[truncated, {len(body.strip()) - len(kept)} more chars]"


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
    assert "GMAIL AUTH FAILED" in result


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
    assert "GMAIL AUTH FAILED" in result


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
    assert "GMAIL AUTH FAILED" in result


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
