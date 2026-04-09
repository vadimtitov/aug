"""Web page fetching tool — extracts readable content from URLs.

Fetches multiple pages simultaneously using async HTTP, then extracts
clean readable text via trafilatura (strips ads, nav, boilerplate).
"""

import asyncio
import logging

import httpx
import trafilatura
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_PAGE = 10_000
_TIMEOUT = 15.0
_HEADERS = {"User-Agent": ("Mozilla/5.0 (compatible; AUG/1.0; +https://github.com/aug)")}


@tool
async def fetch_page(urls: list[str]) -> str:
    """Fetch and extract readable content from one or more web pages simultaneously.

    Use this to read the actual content of a URL — articles, documentation,
    search results pages, etc. Provide multiple URLs to fetch them in parallel.

    Args:
        urls: List of URLs to fetch. All are fetched simultaneously.
    """
    logger.info("fetch_page fetching %d url(s): %s", len(urls), urls)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(*[_fetch_one(client, url) for url in urls])
    return "\n\n---\n\n".join(results)


async def _fetch_one(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a single URL and return extracted readable text."""
    try:
        response = await client.get(url, headers=_HEADERS, follow_redirects=True)
        response.raise_for_status()
        text = trafilatura.extract(
            response.text,
            include_links=True,
            include_tables=True,
            no_fallback=False,
        )
        if not text:
            return f"[{url}]: could not extract readable content."
        if len(text) > _MAX_CHARS_PER_PAGE:
            text = text[:_MAX_CHARS_PER_PAGE] + f"\n... [truncated at {_MAX_CHARS_PER_PAGE} chars]"
        logger.info("fetch_page url=%s extracted %d chars", url, len(text))
        return f"[{url}]\n{text}"
    except httpx.HTTPStatusError as e:
        logger.warning("fetch_page url=%s HTTP %d", url, e.response.status_code, exc_info=True)
        return f"[{url}]: HTTP error {e.response.status_code}."
    except Exception as e:
        logger.warning("fetch_page url=%s error: %s", url, e, exc_info=True)
        return f"[{url}]: failed to fetch — {e}."
