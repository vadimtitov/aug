"""web_fetch — Claude Code's WebFetch contract on AUG's fetching.

The agent passes a URL and a prompt; a small, fast model (the harness's FAST tier) reads
the page and answers the prompt, so the page itself never enters the agent's context.
The page is fetched through aug/utils/fetching (curl_cffi, then the stealth browser), so
bot-protected and JavaScript-rendered pages work.

Like WebFetch: http is upgraded to https, a redirect off the site is returned rather
than followed, a long page is read up to MAX_CONTENT_CHARS at a time with an offset to
read on, and pages are kept for CACHE_TTL_S.  Unlike it: ``raw=True`` returns the page's
own text instead of an answer.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from langchain_core.tools import tool

from aug.core.harness.types import ModelTier
from aug.core.prompts import (
    WEB_FETCH_INSTRUCTIONS,
    WEB_FETCH_PAGE,
    WEB_FETCH_PART_LEAD,
    WEB_FETCH_TRUNCATED,
)
from aug.core.tools.context import current_tool_context
from aug.utils.fetching.fetcher import FetchedPage, browser_client, fetch_url
from aug.utils.fetching.page_content import Classification, full_markdown

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 100_000  # what the small model reads per call
RAW_MAX_CHARS = 40_000  # what raw=True returns per call
CACHE_TTL_S = 15 * 60
MAX_URL_CHARS = 2000

# Reads a prompt, returns the model's answer.
Ask = Callable[[str], Awaitable[str]]


@tool
async def web_fetch(url: str, prompt: str, offset: int = 0, raw: bool = False) -> str:
    """Fetch a URL and answer `prompt` about the page using a small, fast model.

    Reads any web page: articles, docs, product, search and listing pages. Bot-protected
    and JavaScript-rendered sites work (a stealth browser takes over when a plain
    request is blocked). The page is converted to markdown and a small model answers
    the prompt from it, so ask for exactly what you need ("list every pricing plan with
    its price", "what does the function return?"); the result is that model's answer,
    not the page.

    - HTTP is upgraded to HTTPS. A redirect to another host is returned to you rather
      than followed; call again with the redirect URL.
    - A long page is read in parts: when the answer covers only part of the page, a
      note at the end gives the offset to call again with to read on.
    - raw=true returns the page's own text (markdown) instead of an answer, up to
      40,000 characters per call. Use it only when you need the page verbatim.
    - Pages are kept for 15 minutes, so asking again about the same URL is fast.
    - Fails on private or logged-in pages and on hosts without a dot (localhost).

    Args:
        url: The URL to fetch content from.
        prompt: What to find out from the page.
        offset: Character position in the page text to start reading from. Use it to
            read on through a page too long for one call, with the value the previous
            result gave.
        raw: Return the page text itself instead of an answer.
    """
    ctx = current_tool_context()

    async def ask(text: str) -> str:
        if ctx is None:
            raise RuntimeError("no model is available outside a turn")
        return await ctx.oneshot(text, ModelTier.FAST)

    return await read_page(url, prompt, ask, offset=offset, raw=raw)


async def read_page(
    url: str,
    prompt: str,
    ask: Ask,
    *,
    offset: int = 0,
    raw: bool = False,
    max_chars: int = MAX_CONTENT_CHARS,
    instructions: str = WEB_FETCH_INSTRUCTIONS,
) -> str:
    """web_fetch's result for *url*: *ask*'s answer to *prompt* from the page, or the
    page text itself (*raw*), or a redirect, or why the page could not be read."""
    url = url.strip()
    if problem := _url_problem(url):
        return f"Cannot fetch {url!r}: {problem}."
    url = _https(url)
    offset = max(offset, 0)

    page = pages.get(url)
    if page is None:
        page = await fetch_url(url, browser_client(), redirects="same_site")
        if _cacheable(page):
            pages.put(url, page)

    if page.classification is Classification.REDIRECT:
        return _redirect(page, prompt, offset)
    if failure := _failure(page):
        return failure

    # The whole page, as WebFetch converts it: the main-content extraction drops table
    # columns and scrambles inline code.  Not HTML (JSON, plain text): the text as is.
    text = page.text
    if page.html:
        text = await asyncio.to_thread(full_markdown, page.html, page.final_url or url)
    if offset and offset >= len(text):
        return (
            f"Nothing left to read from offset {offset}: this page's text is "
            f"{len(text)} characters long."
        )
    lead = f"[web_fetch note: {page.note}]\n" if page.note else ""
    if raw:
        return lead + _raw(page.url, page.status, text, offset)

    part = text[offset:]
    content = part[:max_chars] + (WEB_FETCH_TRUNCATED if len(part) > max_chars else "")
    request = WEB_FETCH_PAGE.format(content=content, prompt=prompt, instructions=instructions)
    if offset:
        request = WEB_FETCH_PART_LEAD.format(offset=offset, total=len(text)) + request
    started = time.monotonic()
    try:
        answer = await ask(request)
    except Exception as e:
        logger.warning("web_fetch model call failed url=%s: %s", url, e)
        return (
            f"Fetched {url}, but the model that reads the page failed: "
            f"{type(e).__name__}: {e}. The page was NOT read; trying again may work, "
            "or call with raw=true for the page text."
        )
    logger.info(
        "web_fetch url=%s chars=%d offset=%d read=%d answer=%d model=%.1fs",
        url,
        len(text),
        offset,
        min(len(part), max_chars),
        len(answer),
        time.monotonic() - started,
    )
    if not answer.strip():
        return "The small, fast model that reads the content gave no answer. Trying again may work."
    end = offset + min(len(part), max_chars)
    if end < len(text):
        answer += (
            f"\n[web_fetch note: the answer above covers characters {offset} to {end} of "
            f"this page's {len(text)} characters. Next offset: {end}.]"
        )
    return lead + answer


@dataclass(frozen=True)
class _Cached:
    page: FetchedPage
    at: float


class PageCache:
    """Fetched pages by URL, for CACHE_TTL_S: a second question about a page, or reading
    on with an offset, does not fetch it again."""

    def __init__(self, ttl_s: float = CACHE_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._pages: dict[str, _Cached] = {}

    def get(self, url: str) -> FetchedPage | None:
        cached = self._pages.get(url)
        if cached is None or time.monotonic() - cached.at > self._ttl_s:
            return None
        return cached.page

    def put(self, url: str, page: FetchedPage) -> None:
        now = time.monotonic()
        self._pages = {u: c for u, c in self._pages.items() if now - c.at <= self._ttl_s}
        self._pages[url] = _Cached(page, now)

    def clear(self) -> None:
        self._pages.clear()


pages = PageCache()


# --- Private ---


def _url_problem(url: str) -> str | None:
    if len(url) > MAX_URL_CHARS:
        return f"the URL is longer than {MAX_URL_CHARS} characters"
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except ValueError:
        return "not a valid URL"
    if parsed.scheme not in ("http", "https"):
        return "only http and https URLs can be fetched"
    if parsed.username or parsed.password:
        return "URLs with credentials cannot be fetched"
    if "." not in host:
        return "the host has no dot (localhost and local names cannot be fetched)"
    return None


def _https(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(scheme="https")) if parsed.scheme == "http" else url


def _cacheable(page: FetchedPage) -> bool:
    return page.classification is Classification.REDIRECT or (
        page.classification in (Classification.OK, Classification.THIN) and bool(page.text)
    )


def _redirect(page: FetchedPage, prompt: str, offset: int) -> str:
    status = f"{page.status}" if 300 <= page.status < 400 else "unknown (followed in a browser)"
    again = f'\n  - url: "{page.redirect_to}"\n  - prompt: "{prompt}"'
    if offset:
        again += f"\n  - offset: {offset}"
    return (
        "REDIRECT DETECTED: The URL redirects to a location that was not fetched "
        "automatically.\n"
        f"Original URL: {page.final_url or page.url}\n"
        f"Redirect URL: {page.redirect_to}\n"
        f"Status: {status}\n\n"
        "To read the page it redirects to, call web_fetch again with:" + again
    )


def _failure(page: FetchedPage) -> str | None:
    """Why *page* cannot be read, or None when it can."""
    if page.text and page.classification in (Classification.OK, Classification.THIN):
        return None
    status = f"HTTP {page.status}" if page.status else "no response"
    if page.classification is Classification.HTTP_ERROR:
        return f"Failed to fetch {page.url}: {status}. The page content was NOT retrieved."
    why = page.note or page.classification.value
    return f"Failed to fetch {page.url} ({status}): {why}. The page content was NOT retrieved."


def _raw(url: str, status: int, text: str, offset: int) -> str:
    end = min(offset + RAW_MAX_CHARS, len(text))
    shown = f", shown from character {offset}" if offset else ""
    header = f"Fetched {url} (HTTP {status}, {len(text)} characters{shown})"
    body = text[offset:end]
    if end < len(text):
        body += (
            f"\n\n[web_fetch note: showed characters {offset} to {end} of this page's "
            f"{len(text)} characters. Next offset: {end}.]"
        )
    return f"{header}\n\n{body}"
