"""fetch_page — reads web pages, including bot-protected and JavaScript-rendered ones,
and returns their text as markdown.  The fetching itself is aug/utils/fetching."""

import asyncio

from langchain_core.tools import tool

from aug.utils.fetching.fetcher import FetchedPage, browser_client, fetch_url

MAX_CHARS_PER_PAGE = 12_000


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
    browser = browser_client()
    pages = await asyncio.gather(*[fetch_url(url, browser) for url in urls])
    return "\n\n---\n\n".join(render(page, offset) for page in pages)


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
