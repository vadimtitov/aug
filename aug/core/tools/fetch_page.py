"""fetch_page — reads web pages, including bot-protected and JavaScript-rendered ones.

Per URL, two tiers:

- cheap: curl_cffi impersonating Chrome and Safari at once (real browser TLS and HTTP/2
  fingerprints), with the cookies remembered for the domain.  About a second.
- browser: the stealth-browser sidecar (Camoufox), when the cheap tier was blocked by bot
  protection, got a JavaScript shell, or hit a network error worth retrying.  Several
  seconds.  Its cookies are kept, so the cheap tier often works on the domain afterwards.

A genuine HTTP error (404, 410…) is returned as is: a browser would get it too.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal
from urllib.parse import urlparse

from curl_cffi import CurlError
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import (
    DNSError,
    InvalidSchema,
    InvalidURL,
    MissingSchema,
)
from langchain_core.tools import tool

from aug.config import get_settings
from aug.core.tools.page_content import (
    BLOCK_STATUS,
    Classification,
    bot_vendor,
    classify,
    extract,
)
from aug.utils.stealth_browser import (
    BrowserPage,
    StealthBrowserClient,
    StealthBrowserUnavailable,
)

logger = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE = 12_000
_CHEAP_TIMEOUT_S = 15
_MAX_BODY_BYTES = 5_000_000
_DOMAIN_TTL_S = 24 * 3600
# Second-level labels under which a registrable domain takes three labels (bbc.co.uk).
_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "gov", "ac", "edu", "ne", "or", "go"})
_NOT_RETRYABLE = (DNSError, InvalidURL, MissingSchema, InvalidSchema)
_TEXT_TYPES = ("application/json", "application/ld+json", "application/xml", "text/")
_HTML_TYPES = ("text/html", "application/xhtml+xml")

Tier = Literal["cheap", "browser"]


class Strategy(StrEnum):
    CHEAP = "cheap"
    # The browser got through where the cheap tier was blocked; its cookies may let the
    # cheap tier through now.  If they do not, the domain goes browser_first.
    CHEAP_WITH_COOKIES = "cheap_with_cookies"
    BROWSER_FIRST = "browser_first"


@dataclass(frozen=True)
class FetchedPage:
    """One URL's result, whichever tier produced it."""

    url: str
    tier: Tier | None  # None: nothing came back
    classification: Classification
    status: int
    elapsed_s: float
    title: str = ""
    text: str = ""  # the full extracted text; render() picks the window
    note: str | None = None  # why the content is missing or partial
    final_url: str = ""
    retryable: bool = False  # a network error the browser might get past


@dataclass
class _DomainEntry:
    cookies: dict[str, str] = field(default_factory=dict)
    strategy: Strategy = Strategy.CHEAP
    updated_at: float = 0.0


class DomainMemory:
    """Cookies and the strategy that works, per registrable domain, for a day."""

    def __init__(self, ttl_s: float = _DOMAIN_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._entries: dict[str, _DomainEntry] = {}

    def cookies(self, domain: str) -> dict[str, str]:
        return dict(self._entry(domain).cookies)

    def strategy(self, domain: str) -> Strategy:
        return self._entry(domain).strategy

    def add_cookies(self, domain: str, cookies: dict[str, str]) -> None:
        if cookies:
            entry = self._entry(domain)
            entry.cookies.update(cookies)
            entry.updated_at = time.monotonic()

    def set_strategy(self, domain: str, strategy: Strategy) -> None:
        entry = self._entry(domain)
        entry.strategy = strategy
        entry.updated_at = time.monotonic()

    def clear(self) -> None:
        self._entries.clear()

    def _entry(self, domain: str) -> _DomainEntry:
        entry = self._entries.get(domain)
        if entry is None or time.monotonic() - entry.updated_at > self._ttl_s:
            entry = self._entries[domain] = _DomainEntry(updated_at=time.monotonic())
        return entry


domains = DomainMemory()


@tool
async def fetch_page(urls: list[str], offset: int = 0) -> str:
    """Read web pages: fetch one or more URLs in parallel and return their content as markdown.

    This is the tool for reading any web page — articles, docs, product and search pages.
    It handles bot-protected and JavaScript-rendered sites itself (a slower stealth browser
    takes over when a plain request is blocked), so a site that blocks other fetchers
    usually works here.  Each page shows up to 12,000 characters; when a page is longer
    its last line says so and gives the offset to call again with to read on.

    Args:
        urls: URLs to fetch, all at once.
        offset: Character offset into each page's text, to continue a truncated page.
    """
    browser = _browser_client()
    pages = await asyncio.gather(*[fetch_url(url, browser) for url in urls])
    return "\n\n---\n\n".join(render(page, offset) for page in pages)


async def fetch_url(url: str, browser: StealthBrowserClient | None) -> FetchedPage:
    """Fetch *url* through the tiers.  *browser* None: the stealth browser is not configured."""
    started = time.monotonic()
    domain = registrable_domain(url)
    strategy = domains.strategy(domain)
    browser_failure: str | None = None

    if strategy is Strategy.BROWSER_FIRST and browser:
        page, browser_failure = await _browser_tier(url, domain, browser, started)
        if page:
            return _logged(page)

    cheap = await _cheap_tier(url, domain, started)
    if not _escalates(cheap):
        if strategy is Strategy.BROWSER_FIRST and cheap.classification is Classification.OK:
            domains.set_strategy(domain, Strategy.CHEAP)  # the browser failed, cheap works
        return _logged(cheap)
    if strategy is Strategy.CHEAP_WITH_COOKIES:
        domains.set_strategy(domain, Strategy.BROWSER_FIRST)  # the cookies stopped working

    if browser and browser_failure is None:
        page, browser_failure = await _browser_tier(url, domain, browser, started)
        if page:
            if strategy is Strategy.CHEAP:
                domains.set_strategy(
                    domain,
                    Strategy.BROWSER_FIRST
                    if cheap.classification is Classification.THIN
                    else Strategy.CHEAP_WITH_COOKIES,
                )
            return _logged(page)

    reason = browser_failure or "stealth browser unavailable"
    return _logged(replace(cheap, note=_failure_note(cheap, reason)))


def render(page: FetchedPage, offset: int = 0) -> str:
    """The page as the model reads it: a header line, then a window of its text."""
    status = f"HTTP {page.status}" if page.status else "no response"
    header = f"[{page.url}] (via {page.tier or 'nothing'}, {status})"
    if page.final_url and page.final_url.rstrip("/") != page.url.rstrip("/"):
        header += f"\nredirected to {page.final_url}"
    if page.note:
        header += f"\n{page.note}"
    text = page.text
    if not text:
        return header
    total = len(text)
    start = max(offset, 0)
    if start >= total:
        return f"{header}\n[offset {start} is past the end of the page ({total} chars)]"
    end = min(start + MAX_CHARS_PER_PAGE, total)
    body = text[start:end]
    if end < total:
        body += (
            f"\n\n[truncated: showing chars {start}-{end} of {total}; "
            f"call fetch_page again with offset={end}]"
        )
    elif start:
        body += f"\n\n[end of page: showed chars {start}-{end} of {total}]"
    return f"{header}\n{body}"


def registrable_domain(url: str) -> str:
    """The domain a site's cookies and bot protection belong to: www2.hm.com → hm.com."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


# --- Private ---


@dataclass(frozen=True)
class _Response:
    status: int
    final_url: str
    content_type: str
    body: bytes
    size: int  # bytes, from Content-Length when the body was not read
    server: str
    cookies: dict[str, str]
    charset: str | None


async def _cheap_tier(url: str, domain: str, started: float) -> FetchedPage:
    """Chrome and Safari impersonations at once; Chrome's answer wins when both are good."""
    cookies = domains.cookies(domain)
    chrome = asyncio.create_task(_cheap_attempt(url, "chrome", cookies, domain, started))
    safari = asyncio.create_task(_cheap_attempt(url, "safari", cookies, domain, started))
    first = await chrome
    if first.classification is Classification.OK:
        safari.cancel()
        return first
    second = await safari
    return min((first, second), key=lambda p: _CHEAP_PREFERENCE.index(p.classification))


async def _cheap_attempt(
    url: str, impersonate: str, cookies: dict[str, str], domain: str, started: float
) -> FetchedPage:
    try:
        response = await _get(url, impersonate, cookies)
    except CurlError as e:
        return FetchedPage(
            url=url,
            tier="cheap",
            classification=Classification.NETWORK_ERROR,
            status=0,
            elapsed_s=_since(started),
            note=f"network error: {e}",
            retryable=not isinstance(e, _NOT_RETRYABLE),
        )
    except Exception as e:  # one URL's surprise must not fail the others
        logger.exception("fetch_page cheap tier failed url=%s", url)
        return FetchedPage(
            url=url,
            tier="cheap",
            classification=Classification.NETWORK_ERROR,
            status=0,
            elapsed_s=_since(started),
            note=f"fetch failed: {type(e).__name__}: {e}",
            retryable=True,
        )
    domains.add_cookies(domain, response.cookies)
    return _page_from_response(url, response, started)


async def _get(url: str, impersonate: str, cookies: dict[str, str]) -> _Response:
    async with AsyncSession(impersonate=impersonate) as session:
        response = await session.get(
            url,
            cookies=cookies or None,
            timeout=_CHEAP_TIMEOUT_S,
            allow_redirects=True,
            max_redirects=10,
            stream=True,
        )
        try:
            content_type = (response.headers.get("content-type") or "").lower()
            readable = not content_type or content_type.startswith(_HTML_TYPES + _TEXT_TYPES)
            body = b""
            if readable:
                chunks = []
                size = 0
                async for chunk in response.aiter_content():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= _MAX_BODY_BYTES:
                        break
                body = b"".join(chunks)[:_MAX_BODY_BYTES]
            length = response.headers.get("content-length") or ""
            return _Response(
                status=response.status_code,
                final_url=str(response.url),
                content_type=content_type,
                body=body,
                size=len(body) if readable else int(length) if length.isdigit() else 0,
                server=response.headers.get("server") or "",
                cookies=session.cookies.get_dict(),
                charset=response.charset_encoding,
            )
        finally:
            await response.aclose()


def _page_from_response(url: str, response: _Response, started: float) -> FetchedPage:
    content_type = response.content_type
    common = dict(url=url, tier="cheap", status=response.status, final_url=response.final_url)
    if content_type and not content_type.startswith(_HTML_TYPES + _TEXT_TYPES):
        kind = content_type.split(";")[0]
        size = f"{response.size:,} bytes" if response.size else "unknown size"
        return FetchedPage(
            **common,
            classification=_by_status(response.status),
            elapsed_s=_since(started),
            note=f"[binary content: {kind}, {size}; not extracted]",
        )
    text = response.body.decode(response.charset or "utf-8", errors="replace")
    if content_type and not content_type.startswith(_HTML_TYPES):  # JSON, plain text…
        return FetchedPage(
            **common,
            classification=_by_status(response.status),
            elapsed_s=_since(started),
            text=text,
        )
    classification = classify(response.status, text)
    if classification is Classification.BLOCKED:
        vendor = bot_vendor(text, response.server)
        return FetchedPage(
            **common,
            classification=classification,
            elapsed_s=_since(started),
            note=f"blocked by bot protection{f' ({vendor})' if vendor else ''}",
        )
    page = extract(text, response.final_url or url)
    return FetchedPage(
        **common,
        classification=classification,
        elapsed_s=_since(started),
        title=page.title,
        text=page.text,
    )


async def _browser_tier(
    url: str, domain: str, browser: StealthBrowserClient, started: float
) -> tuple[FetchedPage | None, str | None]:
    """The page from the stealth browser, or (None, why it did not get it)."""
    try:
        result = await browser.fetch(url)
    except StealthBrowserUnavailable as e:
        logger.warning("fetch_page stealth browser unavailable url=%s: %s", url, e)
        return None, "stealth browser unavailable"
    _remember_browser_cookies(result)
    if result.outcome == "error":
        return None, f"stealth browser failed: {result.error or 'unknown error'}"
    # The browser judged the page by its content once the challenge was waited out: a
    # 403 left on a document whose challenge was solved in place is not a block.
    solved = result.outcome == "ok" and result.status in BLOCK_STATUS
    classification = classify(200 if solved else result.status, result.html)
    if classification is Classification.BLOCKED:
        vendor = bot_vendor(result.html)
        return None, f"stealth browser also blocked{f' ({vendor})' if vendor else ''}"
    page = extract(result.html, result.final_url or url)
    return (
        FetchedPage(
            url=url,
            tier="browser",
            classification=classification,
            status=result.status,
            elapsed_s=_since(started),
            title=page.title,
            text=page.text,
            final_url=result.final_url,
            note="[page has little text even in a browser]"
            if classification is Classification.THIN
            else None,
        ),
        None,
    )


def _remember_browser_cookies(result: BrowserPage) -> None:
    by_domain: dict[str, dict[str, str]] = {}
    for cookie in result.cookies:
        cookie_domain = registrable_domain(f"https://{cookie.domain.lstrip('.')}/")
        by_domain.setdefault(cookie_domain, {})[cookie.name] = cookie.value
    for cookie_domain, cookies in by_domain.items():
        domains.add_cookies(cookie_domain, cookies)


def _escalates(page: FetchedPage) -> bool:
    """Whether the browser could do better than this cheap-tier result."""
    if page.classification is Classification.NETWORK_ERROR:
        return page.retryable
    return page.classification in (Classification.BLOCKED, Classification.THIN)


def _failure_note(page: FetchedPage, reason: str) -> str:
    if page.classification is Classification.THIN:
        return f"[page needs JavaScript to render; {reason}: the content below may be incomplete]"
    what = page.note or page.classification.value
    return f"[FAILED — {what}; {reason}. The page content was NOT retrieved.]"


def _by_status(status: int) -> Classification:
    """How to judge a response that is not HTML (JSON, plain text, a PDF): by status alone."""
    if status in BLOCK_STATUS:
        return Classification.BLOCKED
    return Classification.HTTP_ERROR if status >= 400 else Classification.OK


def _logged(page: FetchedPage) -> FetchedPage:
    logger.info(
        "fetch_page url=%s tier=%s classification=%s status=%s elapsed=%.2fs chars=%d",
        page.url,
        page.tier,
        page.classification,
        page.status,
        page.elapsed_s,
        len(page.text),
    )
    return page


def _browser_client() -> StealthBrowserClient | None:
    url = get_settings().STEALTH_BROWSER_URL
    return StealthBrowserClient(url) if url else None


def _since(started: float) -> float:
    return round(time.monotonic() - started, 2)


# Which of two failed cheap attempts to keep: a genuine HTTP error is an answer, a
# thin page has some text, a block at least names the vendor.
_CHEAP_PREFERENCE = (
    Classification.OK,
    Classification.HTTP_ERROR,
    Classification.THIN,
    Classification.BLOCKED,
    Classification.NETWORK_ERROR,
)
