"""fetch_page — the web-page reading tool.

tool.py          the tool: tiers (curl_cffi, then the stealth browser), domain memory,
                 rendering for the model
page_content.py  what fetched HTML is (classify) and what it says (extract)
"""

from aug.core.tools.fetch_page.tool import FetchedPage, fetch_page, fetch_url

__all__ = ["FetchedPage", "fetch_page", "fetch_url"]
