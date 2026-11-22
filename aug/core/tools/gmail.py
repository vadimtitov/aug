"""Gmail tools — search, read, send, and draft emails.

Requires a valid OAuth token on disk (run the /auth/gmail flow first).
"""

import logging
import re
from base64 import urlsafe_b64decode, urlsafe_b64encode
from email.mime.text import MIMEText

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool
from lxml import etree
from lxml import html as lxml_html
from markdownify import markdownify

from aug.config import get_settings
from aug.utils.gmail_credentials import load_token, save_token

logger = logging.getLogger(__name__)

_SCOPES = ["https://mail.google.com/"]

# Body characters kept per message: a short preview, and a full read in gmail_read_thread.
_PREVIEW_CHARS = 500
_THREAD_BODY_CHARS = 20_000

_INVISIBLE = re.compile("[\u00ad\u034f\u200b-\u200d\u2060\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_PLACEHOLDER_LINE = re.compile(r"^\[(?:image|header)\]$", re.IGNORECASE)
_BLANK_LINES = re.compile(r"\n{3,}")


class GmailAuthError(Exception):
    """Raised when Gmail credentials are missing or have been revoked."""


def _auth_link(account: str) -> str:
    base = get_settings().base_url
    return f"{base}/auth/gmail?account={account}"


def _is_auth_error(exc: Exception) -> bool:
    if isinstance(exc, (GmailAuthError, RefreshError)):
        return True
    if isinstance(exc, HttpError) and exc.resp.status in (401, 403):
        return True
    return False


def _auth_error_message(account: str) -> str:
    return (
        f"GMAIL AUTH FAILED for account '{account}'. "
        f"The token has expired or been revoked. "
        f"The user MUST re-authorize by visiting: {_auth_link(account)}"
    )


def _get_credentials(account: str) -> Credentials:
    token_data = load_token(account)
    if not token_data:
        raise GmailAuthError(
            f"Gmail account '{account}' is not connected. Authorize here: {_auth_link(account)}"
        )
    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes", _SCOPES),
    )
    if creds.expired and creds.refresh_token:
        logger.info("gmail: refreshing token for account=%r", account)
        creds.refresh(Request())  # raises RefreshError if token revoked — caught by _is_auth_error
        save_token(
            account,
            {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": list(creds.scopes or _SCOPES),
            },
        )
    return creds


def _service(account: str):
    return build("gmail", "v1", credentials=_get_credentials(account), cache_discovery=False)


def _decode_body(payload: dict) -> str:
    """A message's body as text: its text/plain part, else its text/html part as markdown."""
    plain = _find_part(payload, "text/plain")
    if plain.strip():
        return _clean_text(plain)
    html = _find_part(payload, "text/html")
    return _clean_text(_html_to_markdown(html)) if html.strip() else ""


def _format_message(msg: dict, max_chars: int = _PREVIEW_CHARS) -> str:
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "(no subject)")
    sender = headers.get("from", "unknown")
    date = headers.get("date", "")
    snippet = msg.get("snippet", "")
    body = _decode_body(msg.get("payload", {}))
    text = _truncate(body, max_chars) if body else snippet
    return f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{text}"


@tool
async def gmail_search(query: str, max_results: int = 10, account: str = "primary") -> str:
    """Search Gmail messages using Gmail search syntax.

    Supports standard Gmail search operators: from:, to:, subject:, is:unread,
    has:attachment, after:2024/01/01, etc.

    Args:
        query: Gmail search query, e.g. "from:boss@company.com is:unread".
        max_results: Maximum number of messages to return (default 10, max 50).
        account: Gmail account nickname (default "primary").
    """
    try:
        svc = _service(account)
        result = (
            svc.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=min(max_results, 50),
            )
            .execute()
        )
        messages = result.get("messages", [])
        if not messages:
            return f"No messages found for query: {query!r}"
        lines = []
        for m in messages:
            msg = (
                svc.users()
                .messages()
                .get(
                    userId="me",
                    id=m["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])
            }
            lines.append(
                f"[{m['id']}] {headers.get('date', '')} | "
                f"From: {headers.get('from', '?')} | "
                f"Subject: {headers.get('subject', '(no subject)')}"
            )
        return f"Found {len(lines)} message(s):\n" + "\n".join(lines)
    except Exception as e:
        if _is_auth_error(e):
            return _auth_error_message(account)
        logger.exception("gmail_search failed")
        return f"Gmail search failed: {e}"


@tool
async def gmail_read_thread(thread_id: str, account: str = "primary") -> str:
    """Read the full content of a Gmail thread by thread ID.

    Use gmail_search first to find thread IDs, then call this to read the full content.

    Args:
        thread_id: The Gmail thread ID (from gmail_search results).
        account: Gmail account nickname (default "primary").
    """
    try:
        svc = _service(account)
        thread = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
        messages = thread.get("messages", [])
        if not messages:
            return f"Thread {thread_id!r} is empty."
        parts = [f"Thread ID: {thread_id} ({len(messages)} message(s))\n{'=' * 60}"]
        for i, msg in enumerate(messages, 1):
            body = _format_message(msg, max_chars=_THREAD_BODY_CHARS)
            parts.append(f"\n--- Message {i} ---\n{body}")
        return "\n".join(parts)
    except Exception as e:
        if _is_auth_error(e):
            return _auth_error_message(account)
        logger.exception("gmail_read_thread failed")
        return f"Failed to read thread {thread_id!r}: {e}"


@tool
async def gmail_send(to: str, subject: str, body: str, account: str = "primary") -> str:
    """Send an email via Gmail.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        account: Gmail account nickname (default "primary").
    """
    try:
        svc = _service(account)
        mime = MIMEText(body)
        mime["to"] = to
        mime["subject"] = subject
        raw = urlsafe_b64encode(mime.as_bytes()).decode()
        svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        logger.info("gmail_send: sent to=%r subject=%r account=%r", to, subject, account)
        return f"Email sent to {to!r} with subject {subject!r}."
    except Exception as e:
        if _is_auth_error(e):
            return _auth_error_message(account)
        logger.exception("gmail_send failed")
        return f"Failed to send email: {e}"


@tool
async def gmail_draft(to: str, subject: str, body: str, account: str = "primary") -> str:
    """Create a Gmail draft without sending it.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        account: Gmail account nickname (default "primary").
    """
    try:
        svc = _service(account)
        mime = MIMEText(body)
        mime["to"] = to
        mime["subject"] = subject
        raw = urlsafe_b64encode(mime.as_bytes()).decode()
        draft = svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        draft_id = draft.get("id", "unknown")
        logger.info("gmail_draft: created draft_id=%r account=%r", draft_id, account)
        return f"Draft created (ID: {draft_id}) to {to!r} with subject {subject!r}."
    except Exception as e:
        if _is_auth_error(e):
            return _auth_error_message(account)
        logger.exception("gmail_draft failed")
        return f"Failed to create draft: {e}"


def _find_part(payload: dict, mime_type: str) -> str:
    """The decoded data of the first part of this MIME type, searching nested multiparts."""
    part_type = payload.get("mimeType", "")
    if part_type == mime_type:
        data = payload.get("body", {}).get("data", "")
        return urlsafe_b64decode(data + "==").decode("utf-8", errors="replace") if data else ""
    if part_type.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _find_part(part, mime_type)
            if text.strip():
                return text
    return ""


def _html_to_markdown(html: str) -> str:
    try:
        doc = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError):
        return ""
    for node in doc.xpath("//head|//script|//style|//noscript"):
        node.drop_tree()
    return markdownify(
        lxml_html.tostring(doc, encoding="unicode"), heading_style="ATX", strip=["img"]
    )


def _clean_text(text: str) -> str:
    """Collapse the whitespace and drop the image placeholders marketing emails are full of."""
    text = _INVISIBLE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    kept = "\n".join(line for line in lines if not _PLACEHOLDER_LINE.match(line))
    return _BLANK_LINES.sub("\n\n", kept).strip()


def _truncate(text: str, max_chars: int) -> str:
    """At most max_chars of text, cut at a word boundary, saying how much was left out."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    boundary = max(head.rfind(" "), head.rfind("\n"))
    if boundary > max_chars // 2:
        head = head[:boundary]
    head = head.rstrip()
    return f"{head}\n[truncated, {len(text) - len(head)} more chars]"
