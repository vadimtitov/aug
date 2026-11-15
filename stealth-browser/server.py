"""Stealth browser sidecar — fetches a page in Camoufox and returns its HTML and cookies.

AUG's fetch_page calls POST /fetch when a plain request is blocked by bot protection or
the page only renders with JavaScript.  One browser, launched on the first request and
closed after five idle minutes, with one shared context so cookies (bot-protection
clearance included) accumulate across requests.
"""

import asyncio
import contextlib
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Literal

import lxml.html
from camoufox.async_api import AsyncCamoufox
from fastapi import FastAPI
from playwright.async_api import Browser, BrowserContext, Page, Response
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel

logger = logging.getLogger("stealth-browser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_IDLE_CLOSE_S = 300
_MAX_CONCURRENT = 2
_MAX_HTML_BYTES = 5_000_000
_MIN_TEXT = 400
_TRIES = 5
_RELOAD_AFTER_TRY = 2
_BLOCK_STATUS = {401, 403, 405, 429, 503}
# Bot-protection interstitials (Akamai, Cloudflare, DataDome, PerimeterX, Imperva,
# Kasada, AWS WAF).  Keep in step with aug/core/tools/page_content.py.
_CHALLENGE = re.compile(
    r"sec-if-cpt-container|errors\.edgesuite\.net|Just a moment\.\.\.|cf-chl-|"
    r"Attention Required! \| Cloudflare|captcha-delivery\.com|px-captcha|Press &amp; Hold|"
    r"_Incapsula_Resource|Pardon Our Interruption|KPSDK|awsWafCookieDomainList|"
    r"verify you are (a )?human|unusual traffic|Are you a robot|Access Denied|"
    r"Request unsuccessful|bot detection|enable JavaScript and cookies",
    re.I,
)

_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")

Outcome = Literal["ok", "blocked", "error"]


class FetchRequest(BaseModel):
    url: str
    timeout_s: int = 45


class Cookie(BaseModel):
    name: str
    value: str
    domain: str
    path: str
    expires: float


class FetchResponse(BaseModel):
    status: int
    final_url: str
    html: str
    cookies: list[Cookie]
    user_agent: str
    elapsed_s: float
    outcome: Outcome
    error: str | None = None


class StealthBrowser:
    """One lazily launched Camoufox with one shared context; closes itself when idle."""

    def __init__(self) -> None:
        self._camoufox: AsyncCamoufox | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._launch_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(_MAX_CONCURRENT)
        self._in_flight = 0
        self._last_used = time.monotonic()

    @property
    def running(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def fetch(self, url: str, timeout_s: int) -> FetchResponse:
        """Load *url*.  *timeout_s* covers the whole request, waiting for a slot included,
        so a caller that gives up after it is never left with work still running."""
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_s):
                async with self._slots:
                    return await self._fetch_in_slot(url, started)
        except TimeoutError:
            return _error(url, started, f"timed out after {timeout_s}s")

    async def _fetch_in_slot(self, url: str, started: float) -> FetchResponse:
        self._in_flight += 1
        try:
            return await self._fetch(url, started)
        except TimeoutError:
            raise
        except Exception as e:
            logger.exception("fetch failed url=%s", url)
            if not self.running:
                await self.close()  # crashed: the next request relaunches
            return _error(url, started, f"{type(e).__name__}: {e}")
        finally:
            self._in_flight -= 1
            self._last_used = time.monotonic()

    async def close_when_idle(self) -> None:
        """Run forever, closing the browser after _IDLE_CLOSE_S without requests."""
        while True:
            await asyncio.sleep(30)
            idle = time.monotonic() - self._last_used
            if self._browser and not self._in_flight and idle > _IDLE_CLOSE_S:
                logger.info("closing browser after %.0fs idle", idle)
                await self.close()

    async def close(self) -> None:
        async with self._launch_lock:
            camoufox, self._camoufox, self._browser, self._context = (
                self._camoufox,
                None,
                None,
                None,
            )
            if camoufox:
                with contextlib.suppress(Exception):
                    await camoufox.__aexit__(None, None, None)

    async def _fetch(self, url: str, started: float) -> FetchResponse:
        context = await self._ensure_context()
        page = await context.new_page()
        statuses: list[int] = []

        def on_response(response: Response) -> None:
            # The status of the latest main-document response: a challenge that
            # redirects to the real page ends on the real page's status.
            if response.request.is_navigation_request() and response.frame == page.main_frame:
                statuses.append(response.status)

        page.on("response", on_response)
        try:
            response = await page.goto(url, wait_until="domcontentloaded")
            if response and not statuses:
                statuses.append(response.status)
            html, verdict = "", "error"
            for attempt in range(_TRIES):
                await page.wait_for_timeout(1500 if attempt == 0 else 3500)
                html = await _content(page)
                status = statuses[-1] if statuses else 0
                # A challenge solved in place (no new navigation) leaves the 403 on the
                # document; after the first wait, judge such a page by its content.
                if attempt and status in _BLOCK_STATUS:
                    status = 200
                verdict = _judge(status, html)
                if verdict in ("ok", "http_error"):
                    break
                await page.mouse.move(100 + 50 * attempt, 200 + 30 * attempt)
                if attempt == _RELOAD_AFTER_TRY:
                    await page.reload(wait_until="domcontentloaded")
            cookies = await context.cookies([url, page.url])
            user_agent = await page.evaluate("navigator.userAgent")
            outcome: Outcome = "blocked" if verdict in ("blocked", "thin") else "ok"
            logger.info(
                "fetched url=%s status=%s verdict=%s elapsed=%.1fs",
                url,
                statuses[-1] if statuses else 0,
                verdict,
                time.monotonic() - started,
            )
            return FetchResponse(
                status=statuses[-1] if statuses else 0,
                final_url=page.url,
                html=_cap(html),
                cookies=[
                    Cookie(
                        name=c["name"],
                        value=c["value"],
                        domain=c["domain"],
                        path=c["path"],
                        expires=c["expires"],
                    )
                    for c in cookies
                ],
                user_agent=user_agent,
                elapsed_s=round(time.monotonic() - started, 2),
                outcome=outcome,
            )
        finally:
            with contextlib.suppress(Exception):
                await page.close()

    async def _ensure_context(self) -> BrowserContext:
        async with self._launch_lock:
            if self._context is not None and self.running:
                return self._context
            if self._camoufox is not None:  # crashed since the last request
                logger.warning("browser disconnected, relaunching")
                with contextlib.suppress(Exception):
                    await self._camoufox.__aexit__(None, None, None)
            started = time.monotonic()
            self._camoufox = AsyncCamoufox(
                headless=True,
                os="macos",
                geoip=True,
                locale="en-GB",
                humanize=True,
                i_know_what_im_doing=True,
            )
            self._browser = await self._camoufox.__aenter__()
            self._context = await self._browser.new_context()
            logger.info("browser launched in %.1fs", time.monotonic() - started)
            return self._context


_browser = StealthBrowser()


@asynccontextmanager
async def lifespan(_: FastAPI):
    reaper = asyncio.create_task(_browser.close_when_idle())
    yield
    reaper.cancel()
    await _browser.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "browser_running": _browser.running}


@app.post("/fetch")
async def fetch(request: FetchRequest) -> FetchResponse:
    return await _browser.fetch(request.url, request.timeout_s)


# --- Private ---


async def _content(page: Page) -> str:
    """The page's HTML, waiting out a navigation (a challenge redirecting) if one is running."""
    for _ in range(5):
        try:
            return await page.content()
        except PlaywrightError as e:
            if "navigating" not in str(e):
                raise
            await page.wait_for_timeout(1500)
    return await page.content()


def _judge(status: int, html: str) -> str:
    text = _visible_text_len(html)
    challenge = bool(_CHALLENGE.search(html[:30_000])) and text < 3000
    # Same rule as page_content.classify().
    short = text < _MIN_TEXT
    if challenge or (short and (status in _BLOCK_STATUS or status == 202)):
        return "blocked"
    if status >= 400:
        return "http_error"
    if text < _MIN_TEXT:
        return "thin"
    return "ok"


def _visible_text_len(html: str) -> int:
    try:
        # lxml refuses a str that starts with an XML declaration (XHTML pages).
        doc = lxml.html.fromstring(_XML_DECLARATION.sub("", html, count=1))
    except Exception:
        return 0
    for node in doc.xpath("//script|//style|//noscript|//template"):
        node.drop_tree()
    return len(" ".join(doc.text_content().split()))


def _cap(html: str) -> str:
    encoded = html.encode()
    if len(encoded) <= _MAX_HTML_BYTES:
        return html
    return encoded[:_MAX_HTML_BYTES].decode(errors="ignore")


def _error(url: str, started: float, error: str) -> FetchResponse:
    return FetchResponse(
        status=0,
        final_url=url,
        html="",
        cookies=[],
        user_agent="",
        elapsed_s=round(time.monotonic() - started, 2),
        outcome="error",
        error=error,
    )
