"""Web search tool powered by Brave Search API."""

import logging

import httpx
from langchain_core.tools import tool

from aug.config import get_settings

logger = logging.getLogger(__name__)

_API_URL = "https://api.search.brave.com/res/v1/web/search"
_MAX_RESULTS = 5


@tool
def brave_search(query: str) -> str:
    """Search the web using Brave Search and return a summary of the top results.

    Use this when you need up-to-date information or facts you don't know.

    Args:
        query: The search query.
    """
    if not get_settings().BRAVE_API_KEY:
        logger.warning("brave_search called but BRAVE_API_KEY is not set")
        return "Web search is not available: BRAVE_API_KEY is not configured."

    logger.debug("brave_search query=%r", query)
    try:
        response = httpx.get(
            _API_URL,
            headers={
                "X-Subscription-Token": get_settings().BRAVE_API_KEY,
                "Accept": "application/json",
            },
            params={"q": query, "count": _MAX_RESULTS},
            timeout=10.0,
        )
        logger.debug("brave_search status=%d", response.status_code)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("brave_search HTTP error: %s — body: %s", e, e.response.text)
        raise
    except httpx.RequestError as e:
        logger.error("brave_search request failed: %s", e)
        raise

    data = response.json()
    logger.debug("brave_search raw response keys: %s", list(data.keys()))

    results = data.get("web", {}).get("results", [])
    logger.info("brave_search query=%r returned %d results", query, len(results))
    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        description = r.get("description", "")
        lines.append(f"**{title}**\n{url}\n{description}")

    return "\n\n".join(lines)
