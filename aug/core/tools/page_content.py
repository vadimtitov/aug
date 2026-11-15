"""What a fetched web page is (classify) and what it says (extract).

classify() tells a real page from a bot-protection interstitial, a JavaScript shell or an
HTTP error; extract() turns a page's HTML into markdown for the model, with a summary of
any JSON-LD products (retailer pages carry their prices there).
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from html import unescape
from typing import Any
from urllib.parse import urljoin

import lxml.html
import trafilatura
from lxml import etree

# Fewer visible characters than this and the page is a shell that needs JavaScript.
MIN_TEXT = 400

BLOCK_STATUS = frozenset({401, 403, 405, 429, 503})
# Bot-protection interstitials.  Keep in step with stealth-browser/server.py.
_CHALLENGE = re.compile(
    r"sec-if-cpt-container|errors\.edgesuite\.net|Just a moment\.\.\.|cf-chl-|"
    r"Attention Required! \| Cloudflare|captcha-delivery\.com|px-captcha|Press &amp; Hold|"
    r"_Incapsula_Resource|Pardon Our Interruption|KPSDK|awsWafCookieDomainList|"
    r"verify you are (a )?human|unusual traffic|Are you a robot|Access Denied|"
    r"Request unsuccessful|bot detection|enable JavaScript and cookies",
    re.I,
)
# A challenge page is short; a long page that merely mentions "Access Denied" is not one.
_CHALLENGE_MAX_TEXT = 3000
_SCAN_CHARS = 30_000
_VENDORS = (
    ("Akamai", re.compile(r"sec-if-cpt-container|errors\.edgesuite\.net|akamai", re.I)),
    ("Cloudflare", re.compile(r"Just a moment\.\.\.|cf-chl-|cloudflare", re.I)),
    ("DataDome", re.compile(r"captcha-delivery\.com|datadome", re.I)),
    ("PerimeterX", re.compile(r"px-captcha|Press &amp; Hold|perimeterx", re.I)),
    ("Imperva", re.compile(r"_Incapsula_Resource|Pardon Our Interruption|incapsula", re.I)),
    ("Kasada", re.compile(r"KPSDK", re.I)),
    ("AWS WAF", re.compile(r"awsWafCookieDomainList|aws-waf", re.I)),
)
# Markdown shorter than this, or than this share of the visible text, missed the page
# (product listings): use the visible text instead.
_MIN_MARKDOWN = 500
_MIN_MARKDOWN_SHARE = 0.15
_MAX_PRODUCTS = 10
# A markdown link; its text may hold one level of brackets (Wikipedia's "[[54]](…)").
_LINK = re.compile(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)\)")
# An in-page anchor as trafilatura writes it, bare or resolved to the host alone.
_ANCHOR = re.compile(r"^(?:https?://[^/#\s]+/?)?#")
_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")


class Classification(StrEnum):
    OK = "ok"
    BLOCKED = "blocked"  # bot protection
    THIN = "thin"  # too little text: a JavaScript-rendered shell
    HTTP_ERROR = "http_error"  # a genuine 404 and the like; a browser would get it too
    NETWORK_ERROR = "network_error"


@dataclass(frozen=True)
class PageText:
    title: str
    text: str  # markdown: title, product summary, body


def classify(status: int, html: str, visible: str | None = None) -> Classification:
    """What an HTTP response is, from its status and HTML.

    *visible*: the page's visible_text(), when the caller already has it.
    """
    text_len = len(visible_text(html) if visible is None else visible)
    challenge = bool(_CHALLENGE.search(html[:_SCAN_CHARS])) and text_len < _CHALLENGE_MAX_TEXT
    # A blocking status on a page with next to no text is bot protection; with a real
    # page behind it (a login or maintenance page) it is a genuine error.  AWS WAF
    # answers a bot with an empty 202 and a JavaScript challenge.
    short = text_len < MIN_TEXT
    if challenge or (short and (status in BLOCK_STATUS or status == 202)):
        return Classification.BLOCKED
    if status >= 400:
        return Classification.HTTP_ERROR
    if text_len < MIN_TEXT:
        return Classification.THIN
    return Classification.OK


def bot_vendor(html: str, server_header: str = "") -> str | None:
    """The bot-protection vendor behind a blocked page, if it can be told."""
    haystack = f"{server_header}\n{html[:_SCAN_CHARS]}"
    return next((name for name, pattern in _VENDORS if pattern.search(haystack)), None)


def extract(html: str, url: str, visible: str | None = None) -> PageText:
    """The page as markdown: '# title', a JSON-LD product summary, then the body.

    *visible*: the page's visible_text(), when the caller already has it.
    """
    title = page_title(html)
    if visible is None:
        visible = visible_text(html)
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        favor_recall=True,
    )
    body = _absolute_links(markdown or "", url)
    missed = len(body) < _MIN_MARKDOWN or len(body) < _MIN_MARKDOWN_SHARE * len(visible)
    if missed and len(visible) > len(body):
        body = visible
    parts = [f"# {title}"] if title else []
    if products := product_summary(html):
        parts.append(products)
    if body:
        parts.append(body)
    return PageText(title=title, text="\n\n".join(parts))


def page_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if not match:
        return ""
    return " ".join(unescape(re.sub(r"<[^>]+>", "", match.group(1))).split())


def visible_text(html: str) -> str:
    """The page's text without scripts, styles and templates, whitespace collapsed."""
    if not html.strip():
        return ""
    doc = _parse(html)
    if doc is None:
        return ""
    for node in doc.xpath("//script|//style|//noscript|//template"):
        node.drop_tree()
    return " ".join(" ".join(doc.itertext()).split())


def product_summary(html: str) -> str:
    """A compact list of the JSON-LD Products on the page (name, price, availability)."""
    doc = _parse(html)
    if doc is None:
        return ""
    lines: list[str] = []
    for script in doc.xpath('//script[@type="application/ld+json"]'):
        try:
            data = json.loads(script.text_content())
        except ValueError:
            continue
        for product in _products(data):
            line = _product_line(product)
            if line and line not in lines:
                lines.append(line)
    if not lines:
        return ""
    shown = lines[:_MAX_PRODUCTS]
    more = f"\n- … and {len(lines) - len(shown)} more" if len(lines) > len(shown) else ""
    return "Products (JSON-LD):\n" + "\n".join(f"- {line}" for line in shown) + more


# --- Private ---


def _parse(html: str) -> lxml.html.HtmlElement | None:
    # lxml refuses a str that starts with an XML declaration (XHTML pages).
    try:
        return lxml.html.fromstring(_XML_DECLARATION.sub("", html, count=1))
    except (etree.LxmlError, ValueError):
        return None


def _absolute_links(markdown: str, url: str) -> str:
    """Relative links made absolute; in-page anchors (noise, e.g. citations) reduced to text."""

    def fix(match: re.Match) -> str:
        text, target = match.group(1), match.group(2)
        if _ANCHOR.match(target):
            return text
        return f"[{text}]({urljoin(url, target)})"

    return _LINK.sub(fix, markdown)


def _products(data: Any) -> list[dict]:
    """Every Product in a JSON-LD document, however it is nested (@graph, ItemList…)."""
    if isinstance(data, list):
        return [p for item in data for p in _products(item)]
    if not isinstance(data, dict):
        return []
    types = data.get("@type")
    types = types if isinstance(types, list) else [types]
    if "Product" in types:
        return [data]
    if "ProductGroup" in types:  # its variants carry the prices (H&M)
        variants = [] if "offers" in data else _products(data.get("hasVariant", []))
        return variants or [data]
    found: list[dict] = []
    for key in ("@graph", "itemListElement", "item", "mainEntity", "hasVariant"):
        if key in data:
            found.extend(_products(data[key]))
    return found


def _product_line(product: dict) -> str:
    name = product.get("name")
    if not isinstance(name, str) or not name.strip():
        return ""
    fields = [" ".join(name.split())]
    if isinstance(size := product.get("size"), str) and size:
        fields[0] += f" (size {size})"
    offers = product.get("offers")
    offer = offers[0] if isinstance(offers, list) and offers else offers
    if isinstance(offer, dict):
        price = offer.get("price") or offer.get("lowPrice")
        if price is not None and offer.get("highPrice") and offer.get("highPrice") != price:
            price = f"{price}-{offer['highPrice']}"
        if price is not None:
            fields.append(f"{price} {offer.get('priceCurrency', '')}".strip())
        availability = offer.get("availability")
        if isinstance(availability, str) and availability:
            fields.append(availability.rsplit("/", 1)[-1])
    return " — ".join(fields)
