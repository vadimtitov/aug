"""Client for the stealth-browser sidecar (stealth-browser/server.py).

The sidecar loads a page in Camoufox, waits out bot-protection challenges and returns
the rendered HTML with the browser's cookies.
"""

from dataclasses import dataclass
from typing import Literal

import httpx

BrowserOutcome = Literal["ok", "blocked", "error"]


class StealthBrowserUnavailable(Exception):
    """The sidecar could not be reached or answered with something other than a page."""


@dataclass(frozen=True)
class BrowserCookie:
    name: str
    value: str
    domain: str
    path: str
    expires: float


@dataclass(frozen=True)
class BrowserPage:
    status: int
    final_url: str
    html: str
    cookies: list[BrowserCookie]
    user_agent: str
    elapsed_s: float
    outcome: BrowserOutcome
    error: str | None


class StealthBrowserClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def fetch(self, url: str, *, page_timeout_s: int = 45) -> BrowserPage:
        """Load *url* in the browser.  Raises StealthBrowserUnavailable if the sidecar fails."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/fetch", json={"url": url, "timeout_s": page_timeout_s}
                )
                response.raise_for_status()
                data = response.json()
            return BrowserPage(
                status=int(data["status"]),
                final_url=data["final_url"],
                html=data["html"],
                cookies=[BrowserCookie(**c) for c in data["cookies"]],
                user_agent=data["user_agent"],
                elapsed_s=float(data["elapsed_s"]),
                outcome=data["outcome"],
                error=data.get("error"),
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            raise StealthBrowserUnavailable(f"{type(e).__name__}: {e}") from e
