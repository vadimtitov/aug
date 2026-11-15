"""Page fetching and the fetch_page tool: classification, tier escalation, domain memory,
extraction, redirects and rendering.

The cheap tier is faked at curl_cffi's AsyncSession, the browser tier at the
StealthBrowserClient (and the client itself at httpx's transport).  No network.
"""

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
import pytest
from curl_cffi.requests.exceptions import DNSError, Timeout

from aug.core.tools import fetch_page as tool_module
from aug.core.tools.fetch_page import MAX_CHARS_PER_PAGE, render
from aug.utils.fetching import fetcher as fp
from aug.utils.fetching.fetcher import (
    FetchedPage,
    Strategy,
    domains,
    fetch_url,
    is_same_site,
    registrable_domain,
    url_problem,
)
from aug.utils.fetching.page_content import (
    Classification,
    bot_vendor,
    classify,
    extract,
    full_markdown,
    product_summary,
)
from aug.utils.stealth_browser import (
    BrowserCookie,
    BrowserPage,
    StealthBrowserClient,
    StealthBrowserUnavailable,
)

_ARTICLE = "<html><head><title>Real page</title></head><body><article>{}</article></body></html>"
_GOOD_HTML = _ARTICLE.format(
    "".join(
        f"<p>Paragraph {i} about sweatpants and their many fine qualities.</p>" for i in range(30)
    )
)
_AKAMAI_HTML = (
    "<html><head><title>Access Denied</title></head><body>"
    '<div id="sec-if-cpt-container">Please wait</div>'
    "Reference errors.edgesuite.net</body></html>"
)
_CLOUDFLARE_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    '<div class="cf-chl-widget">Checking your browser</div></body></html>'
)
_SHELL_HTML = '<html><head><title>App</title></head><body><div id="root"></div></body></html>'


@pytest.fixture(autouse=True)
def _fresh_domains():
    domains.clear()
    yield
    domains.clear()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Reply:
    status: int = 200
    body: str = _GOOD_HTML
    content_type: str = "text/html; charset=utf-8"
    set_cookies: dict[str, str] = field(default_factory=dict)
    error: Exception | None = None
    headers: dict[str, str] = field(default_factory=dict)
    charset: str = "utf-8"
    delay_s: float = 0.0
    redirects: dict[str, tuple[int, str]] = field(default_factory=dict)  # url → (status, to)


class _FakeCurl:
    """Stands in for curl_cffi's AsyncSession; replies per impersonation target."""

    def __init__(self, replies: dict[str, _Reply]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, impersonate: str) -> "_FakeSession":
        return _FakeSession(self, impersonate)


class _FakeCookies:
    def __init__(self, cookies: dict[str, str]) -> None:
        self._cookies = cookies

    def get_dict(self) -> dict[str, str]:
        return dict(self._cookies)


class _FakeResponse:
    def __init__(self, url: str, reply: _Reply) -> None:
        self.url = url
        self.status_code = reply.status
        self.headers = {"content-type": reply.content_type, **reply.headers}
        self.charset_encoding = reply.charset
        self._body = reply.body.encode()

    async def aiter_content(self):
        yield self._body

    async def aclose(self) -> None:
        pass


class _FakeSession:
    def __init__(self, curl: _FakeCurl, impersonate: str) -> None:
        self._curl = curl
        self._impersonate = impersonate
        self.cookies = _FakeCookies({})

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> None:
        pass

    async def get(self, url: str, *, cookies: dict | None = None, **_) -> _FakeResponse:
        self._curl.calls.append((self._impersonate, url, cookies))
        reply = self._curl.replies[self._impersonate]
        await asyncio.sleep(reply.delay_s)
        if reply.error:
            raise reply.error
        self.cookies = _FakeCookies({**(cookies or {}), **reply.set_cookies})
        if url in reply.redirects:
            status, location = reply.redirects[url]
            return _FakeResponse(
                url, _Reply(status=status, body="", headers={"location": location})
            )
        return _FakeResponse(url, reply)


class _FakeBrowser:
    def __init__(self, page: BrowserPage | None = None, *, down: bool = False) -> None:
        self.page = page
        self.down = down
        self.calls: list[str] = []

    async def fetch(self, url: str, *, page_timeout_s: int = 45) -> BrowserPage:
        self.calls.append(url)
        if self.down:
            raise StealthBrowserUnavailable("ConnectError: connection refused")
        assert self.page is not None
        return self.page


def _browser_page(
    html: str = _GOOD_HTML,
    status: int = 200,
    cookies: list[BrowserCookie] | None = None,
    final_url: str = "https://www2.hm.com/en_gb/p.html",
) -> BrowserPage:
    return BrowserPage(
        status=status,
        final_url=final_url,
        html=html,
        cookies=cookies
        if cookies is not None
        else [BrowserCookie("_abck", "clearance", ".hm.com", "/", 0)],
        user_agent="Mozilla/5.0 Firefox",
        elapsed_s=6.0,
        outcome="ok",
        error=None,
    )


def _curl(chrome: _Reply, safari: _Reply | None = None) -> _FakeCurl:
    return _FakeCurl({"chrome": chrome, "safari": safari or chrome})


_URL = "https://www2.hm.com/en_gb/p.html"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 405, 429, 503])
def test_a_blocking_status_with_no_page_behind_it_is_blocked(status):
    assert classify(status, _SHELL_HTML) is Classification.BLOCKED


@pytest.mark.parametrize("status", [401, 403, 405, 429, 503])
def test_a_blocking_status_on_a_short_page_with_some_text_is_still_blocked(status):
    # Akamai's block page: a reference number and a list of languages.
    assert classify(status, _GOOD_HTML) is Classification.BLOCKED


@pytest.mark.parametrize("status", [401, 403, 405, 429, 503])
def test_a_blocking_status_on_a_substantial_page_is_a_genuine_error(status):
    # A full login or maintenance page: the browser would get the same, the model reads it.
    long_page = _ARTICLE.format("<p>Our service is down for maintenance today.</p>" * 80)
    assert classify(status, long_page) is Classification.HTTP_ERROR


def test_an_xhtml_page_with_an_xml_declaration_is_read():
    xhtml = '<?xml version="1.0" encoding="UTF-8"?>\n' + _GOOD_HTML
    assert classify(200, xhtml) is Classification.OK
    assert "Paragraph 29" in extract(xhtml, "https://example.com/").text


def test_akamai_and_cloudflare_challenges_are_blocked_with_their_vendor():
    assert classify(200, _AKAMAI_HTML) is Classification.BLOCKED
    assert classify(200, _CLOUDFLARE_HTML) is Classification.BLOCKED
    assert bot_vendor(_AKAMAI_HTML) == "Akamai"
    assert bot_vendor(_CLOUDFLARE_HTML) == "Cloudflare"
    assert bot_vendor("<html></html>", server_header="AkamaiGHost") == "Akamai"


def test_aws_waf_empty_202_is_blocked():
    assert classify(202, "<html><body></body></html>") is Classification.BLOCKED


def test_a_long_page_that_mentions_access_denied_is_not_a_challenge():
    html = _ARTICLE.format("Access Denied is a film. " + "Plot summary goes on. " * 200)
    assert classify(200, html) is Classification.OK


def test_shell_is_thin_404_is_http_error_article_is_ok():
    assert classify(200, _SHELL_HTML) is Classification.THIN
    assert classify(404, _GOOD_HTML) is Classification.HTTP_ERROR
    assert classify(200, _GOOD_HTML) is Classification.OK


def test_registrable_domain():
    assert registrable_domain("https://www2.hm.com/x") == "hm.com"
    assert registrable_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert registrable_domain("https://en.wikipedia.org/wiki/A") == "wikipedia.org"


def test_sites_on_a_shared_host_and_ip_addresses_are_kept_apart():
    assert registrable_domain("https://alice.github.io/") == "alice.github.io"
    assert registrable_domain("https://bob.github.io/") == "bob.github.io"
    assert registrable_domain("http://10.0.1.10:8000/") == "10.0.1.10"
    assert registrable_domain("http://192.168.1.10/") == "192.168.1.10"
    assert registrable_domain("http://localhost:3000/") == "localhost"


def test_domain_memory_lasts_a_day_from_the_first_fetch_however_often_it_is_used():
    clock = [1000.0]
    with patch.object(fp.time, "monotonic", lambda: clock[0]):
        memory = fp.DomainMemory(ttl_s=100)
        memory.set_strategy("hm.com", Strategy.BROWSER_FIRST)
        memory.add_cookies("other.com", {"a": "1"})
        for _ in range(3):
            clock[0] += 40
            memory.add_cookies("hm.com", {"b": "2"})  # use does not extend it

        assert memory.strategy("hm.com") is Strategy.CHEAP
        assert "other.com" not in memory._entries  # expired domains are dropped


# ---------------------------------------------------------------------------
# Escalation and domain memory
# ---------------------------------------------------------------------------


async def test_ok_cheap_page_never_touches_the_browser():
    curl, browser = _curl(_Reply()), _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url(_URL, browser)

    assert page.tier == "cheap" and page.classification is Classification.OK
    assert "# Real page" in page.text
    assert browser.calls == []


async def test_404_is_returned_without_escalating():
    curl, browser = _curl(_Reply(status=404)), _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url(_URL, browser)

    assert page.classification is Classification.HTTP_ERROR and page.status == 404
    assert browser.calls == []
    assert "HTTP 404" in render(page)


async def test_blocked_escalates_stores_cookies_and_next_call_uses_them_on_the_cheap_path():
    browser = _FakeBrowser(_browser_page())
    blocked = _curl(_Reply(status=403, body=_AKAMAI_HTML))
    with patch.object(fp, "AsyncSession", blocked):
        page = await fetch_url(_URL, browser)

    assert page.tier == "browser" and page.classification is Classification.OK
    assert browser.calls == [_URL]
    assert domains.cookies("hm.com") == {"_abck": "clearance"}
    assert domains.strategy("hm.com") is Strategy.CHEAP_WITH_COOKIES

    now_ok = _curl(_Reply())
    with patch.object(fp, "AsyncSession", now_ok):
        second = await fetch_url("https://www2.hm.com/en_gb/other.html", browser)

    assert second.tier == "cheap"
    assert browser.calls == [_URL]  # no second browser trip
    assert all(cookies == {"_abck": "clearance"} for _, _, cookies in now_ok.calls)


async def test_cookies_that_stop_working_send_the_domain_browser_first():
    browser = _FakeBrowser(_browser_page())
    blocked = _curl(_Reply(status=403, body=_AKAMAI_HTML))
    with patch.object(fp, "AsyncSession", blocked):
        await fetch_url(_URL, browser)  # → cheap_with_cookies
        await fetch_url(_URL, browser)  # cheap still blocked → browser_first

    assert domains.strategy("hm.com") is Strategy.BROWSER_FIRST
    blocked.calls.clear()
    with patch.object(fp, "AsyncSession", blocked):
        page = await fetch_url(_URL, browser)

    assert page.tier == "browser"
    assert blocked.calls == []  # straight to the browser


async def test_thin_page_goes_browser_first_next_time():
    browser = _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", _curl(_Reply(body=_SHELL_HTML))):
        page = await fetch_url("https://spa.example.com/", browser)

    assert page.tier == "browser"
    assert domains.strategy("example.com") is Strategy.BROWSER_FIRST


async def test_a_page_short_even_in_the_browser_leaves_the_domain_on_the_cheap_path():
    browser = _FakeBrowser(_browser_page(html=_SHELL_HTML))
    with patch.object(fp, "AsyncSession", _curl(_Reply(body=_SHELL_HTML))):
        page = await fetch_url("https://spa.example.com/", browser)

    assert page.tier == "browser" and page.classification is Classification.THIN
    assert domains.strategy("example.com") is Strategy.CHEAP


async def test_browser_down_returns_an_explicit_failure():
    browser = _FakeBrowser(down=True)
    blocked = _curl(_Reply(status=403, body=_AKAMAI_HTML))
    with patch.object(fp, "AsyncSession", blocked):
        page = await fetch_url(_URL, browser)

    out = render(page)
    assert "blocked by bot protection (Akamai)" in out
    assert "stealth browser unavailable" in out
    assert "NOT retrieved" in out
    assert domains.strategy("hm.com") is Strategy.CHEAP


async def test_no_browser_configured_returns_an_explicit_failure():
    with patch.object(fp, "AsyncSession", _curl(_Reply(status=403, body=_CLOUDFLARE_HTML))):
        page = await fetch_url(_URL, None)

    assert "[FAILED — blocked by bot protection (Cloudflare); stealth browser unavailable" in (
        render(page)
    )


async def test_browser_also_blocked_says_so():
    browser = _FakeBrowser(_browser_page(html=_AKAMAI_HTML, status=403))
    with patch.object(fp, "AsyncSession", _curl(_Reply(status=403, body=_AKAMAI_HTML))):
        page = await fetch_url(_URL, browser)

    assert "stealth browser also blocked (Akamai)" in render(page)


async def test_thin_without_browser_returns_what_there_is_with_a_note():
    with patch.object(fp, "AsyncSession", _curl(_Reply(body=_SHELL_HTML))):
        page = await fetch_url("https://spa.example.com/", None)

    out = render(page)
    assert "needs JavaScript" in out and "# App" in out


async def test_safari_wins_when_chrome_is_blocked():
    curl = _curl(_Reply(status=403, body=_AKAMAI_HTML), _Reply())
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url(_URL, None)

    assert page.tier == "cheap" and page.classification is Classification.OK


async def test_a_good_safari_answer_does_not_wait_for_a_slow_chrome():
    curl = _curl(_Reply(delay_s=5), _Reply())
    with patch.object(fp, "AsyncSession", curl):
        page = await asyncio.wait_for(fetch_url(_URL, None), timeout=1)

    assert page.classification is Classification.OK


async def test_dns_failure_is_not_retried_in_the_browser_but_timeout_is():
    browser = _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", _curl(_Reply(error=DNSError("could not resolve")))):
        page = await fetch_url("https://nope.invalid/", browser)
    assert page.classification is Classification.NETWORK_ERROR
    assert browser.calls == []
    assert "network error" in render(page)

    with patch.object(fp, "AsyncSession", _curl(_Reply(error=Timeout("timed out")))):
        page = await fetch_url("https://slow.example.com/", browser)
    assert page.tier == "browser"


async def test_json_is_returned_raw_and_binary_as_a_notice():
    with patch.object(
        fp, "AsyncSession", _curl(_Reply(body='{"a": 1}', content_type="application/json"))
    ):
        page = await fetch_url("https://api.example.com/x", None)
    assert page.classification is Classification.OK and page.text == '{"a": 1}'

    pdf = _Reply(body="", content_type="application/pdf", headers={"content-length": "2048"})
    with patch.object(fp, "AsyncSession", _curl(pdf)):
        page = await fetch_url("https://example.com/a.pdf", None)
    assert "binary content: application/pdf, 2,048 bytes" in render(page)


@pytest.mark.parametrize(
    "content_type", ["application/rss+xml", "application/vnd.api+json", "text/csv"]
)
async def test_feeds_and_structured_text_are_returned_as_text(content_type):
    with patch.object(fp, "AsyncSession", _curl(_Reply(body="<rss/>", content_type=content_type))):
        page = await fetch_url("https://blog.example.com/feed", None)
    assert page.text == "<rss/>"


async def test_an_apis_own_error_is_returned_for_the_model_to_read():
    reply = _Reply(status=401, body='{"message": "Requires auth"}', content_type="application/json")
    browser = _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", _curl(reply)):
        page = await fetch_url("https://api.example.com/x", browser)

    assert page.classification is Classification.HTTP_ERROR
    assert "Requires auth" in render(page)
    assert browser.calls == []


async def test_an_unknown_charset_is_read_as_utf8():
    with patch.object(fp, "AsyncSession", _curl(_Reply(charset="utf8mb4"))):
        page = await fetch_url(_URL, None)
    assert page.classification is Classification.OK


def _fails_on_a_com(html: str, url: str, visible: str | None = None):
    if "a.com" in url:
        raise RuntimeError("boom")
    return extract(html, url, visible)


async def test_a_failure_on_one_url_leaves_the_others_in_the_tool_output():
    with (
        patch.object(fp, "AsyncSession", _curl(_Reply())),
        patch.object(tool_module, "browser_client", return_value=None),
        patch.object(fp, "extract", side_effect=_fails_on_a_com),
    ):
        out = await tool_module.fetch_page.ainvoke({"urls": ["https://a.com/", "https://b.com/"]})

    assert "RuntimeError: boom" in out and "NOT retrieved" in out
    assert "Paragraph 29" in out


async def test_the_tool_fetches_urls_in_parallel_and_joins_them():
    with (
        patch.object(fp, "AsyncSession", _curl(_Reply())),
        patch.object(tool_module, "browser_client", return_value=None),
    ):
        out = await tool_module.fetch_page.ainvoke({"urls": ["https://a.com/", "https://b.com/"]})

    assert "[https://a.com/] (via cheap, HTTP 200)" in out
    assert "[https://b.com/] (via cheap, HTTP 200)" in out
    assert "\n\n---\n\n" in out


# ---------------------------------------------------------------------------
# Extraction and rendering
# ---------------------------------------------------------------------------


def test_json_ld_product_summary():
    ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Product",
                "name": "Relaxed Fit Joggers",
                "offers": {
                    "@type": "Offer",
                    "price": "24.99",
                    "priceCurrency": "GBP",
                    "availability": "https://schema.org/InStock",
                },
            }
        ],
    }
    html = _GOOD_HTML.replace(
        "</head>", f'<script type="application/ld+json">{json.dumps(ld)}</script></head>'
    )

    text = extract(html, "https://shop.example.com/p").text

    assert text.startswith("# Real page")
    assert "Products (JSON-LD):\n- Relaxed Fit Joggers — 24.99 GBP — InStock" in text


def test_product_group_lists_its_variants_with_sizes():
    ld = {
        "@type": "ProductGroup",
        "name": "Sweatpants",
        "hasVariant": [
            {
                "@type": "Product",
                "name": "Sweatpants - Black",
                "size": size,
                "offers": {"price": 19.99, "priceCurrency": "GBP", "availability": stock},
            }
            for size, stock in (("S", "https://schema.org/InStock"), ("M", "OutOfStock"))
        ],
    }
    html = f'<html><script type="application/ld+json">{json.dumps(ld)}</script></html>'

    assert product_summary(html) == (
        "Products (JSON-LD):\n"
        "- Sweatpants - Black (size S) — 19.99 GBP — InStock\n"
        "- Sweatpants - Black (size M) — 19.99 GBP — OutOfStock"
    )


def test_listing_falls_back_to_visible_text_when_markdown_misses_it():
    tiles = "".join(
        f'<li><a href="/p{i}">Jogger {i}</a><span>£{i}.99</span></li>' for i in range(80)
    )
    html = f"<html><head><title>Joggers</title></head><body><ul>{tiles}</ul></body></html>"

    text = extract(html, "https://shop.example.com/c").text

    assert "Jogger 79" in text and "£79.99" in text


def test_render_truncates_and_points_at_the_next_offset():
    page = FetchedPage(
        url="https://x.com/", tier="cheap", classification=Classification.OK, status=200,
        elapsed_s=1.0, text="a" * 30_000,
    )  # fmt: skip

    first = render(page)
    assert first.startswith("[https://x.com/] (via cheap, HTTP 200)")
    assert (
        f"[truncated: showing chars 0-{MAX_CHARS_PER_PAGE} of 30000; "
        f"call fetch_page again with offset={MAX_CHARS_PER_PAGE}]"
    ) in first

    last = render(page, offset=24_000)
    assert "[end of page: showed chars 24000-30000 of 30000]" in last
    assert "past the end" in render(page, offset=40_000)


# ---------------------------------------------------------------------------
# Stealth browser client
# ---------------------------------------------------------------------------


async def test_client_parses_the_sidecar_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/fetch"
        assert json.loads(request.content) == {"url": _URL, "timeout_s": 45}
        return httpx.Response(
            200,
            json={
                "status": 200,
                "final_url": _URL,
                "html": "<html></html>",
                "cookies": [
                    {"name": "a", "value": "b", "domain": ".hm.com", "path": "/", "expires": -1}
                ],
                "user_agent": "UA",
                "elapsed_s": 5.5,
                "outcome": "ok",
                "error": None,
            },
        )

    client = StealthBrowserClient("http://sb:8100/", transport=httpx.MockTransport(handler))
    page = await client.fetch(_URL)

    assert page.status == 200 and page.cookies == [BrowserCookie("a", "b", ".hm.com", "/", -1)]


async def test_client_raises_unavailable_on_connection_errors_and_bad_replies():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(StealthBrowserUnavailable):
        await StealthBrowserClient("http://sb", transport=httpx.MockTransport(refused)).fetch(_URL)

    broken = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(StealthBrowserUnavailable):
        await StealthBrowserClient("http://sb", transport=broken).fetch(_URL)


async def test_browser_first_domain_goes_back_to_cheap_when_the_browser_fails_and_cheap_works():
    domains.set_strategy("hm.com", Strategy.BROWSER_FIRST)
    with patch.object(fp, "AsyncSession", _curl(_Reply())):
        page = await fetch_url(_URL, _FakeBrowser(down=True))

    assert page.tier == "cheap"
    assert domains.strategy("hm.com") is Strategy.CHEAP


async def test_an_unexpected_error_on_one_url_does_not_fail_the_others():
    curl = _curl(_Reply(error=RuntimeError("boom")))
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url("https://x.example.com/", None)

    assert page.classification is Classification.NETWORK_ERROR
    assert "fetch failed: RuntimeError: boom" in render(page)


def test_in_page_anchor_links_are_reduced_to_their_text():
    html = _ARTICLE.format(
        '<p>See <a href="#cite-1">[1]</a> and <a href="/wiki/Other">Other</a>.</p>' * 20
    )

    text = extract(html, "https://en.wikipedia.org/wiki/A").text

    assert "See [1] and [Other](https://en.wikipedia.org/wiki/Other)" in text
    assert "#cite-1" not in text


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "target", "same"),
    [
        ("https://example.com/a", "https://example.com/b", True),
        ("https://example.com/a", "https://www.example.com/b", True),
        ("https://www.example.com/a", "https://example.com/b", True),
        ("https://example.com/a", "https://docs.example.com/b", False),
        ("https://example.com/a", "https://other.com/", False),
        ("https://example.com/a", "http://example.com/a", False),
        ("https://example.com/a", "https://example.com:8443/a", False),
        ("https://example.com/a", "https://user:pw@example.com/a", False),
    ],
)
def test_is_same_site(url, target, same):
    assert is_same_site(url, target) is same


async def test_same_site_redirects_are_followed_by_hand():
    reply = _Reply(redirects={"https://www.a.com/": (301, "https://a.com/home")})
    curl = _curl(reply)
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url("https://www.a.com/", None, redirects="same_site")

    assert page.classification is Classification.OK
    assert page.final_url == "https://a.com/home"
    assert "Paragraph 29" in page.text


async def test_a_redirect_off_the_site_is_returned_not_followed():
    reply = _Reply(redirects={"https://bit.ly/x": (302, "https://shop.com/item?id=1")})
    curl = _curl(reply)
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url("https://bit.ly/x", _FakeBrowser(), redirects="same_site")

    assert page.classification is Classification.REDIRECT
    assert page.redirect_to == "https://shop.com/item?id=1"
    assert page.status == 302 and not page.text
    assert all(url == "https://bit.ly/x" for _, url, _ in curl.calls)  # never fetched


async def test_a_relative_location_resolves_against_the_url():
    reply = _Reply(redirects={"https://a.com/old": (308, "/new")})
    with patch.object(fp, "AsyncSession", _curl(reply)):
        page = await fetch_url("https://a.com/old", None, redirects="same_site")

    assert page.final_url == "https://a.com/new"


async def test_a_redirect_loop_is_a_network_error():
    reply = _Reply(redirects={"https://a.com/": (302, "https://a.com/")})
    with patch.object(fp, "AsyncSession", _curl(reply)):
        page = await fetch_url("https://a.com/", None, redirects="same_site")

    assert page.classification is Classification.NETWORK_ERROR
    assert "redirects" in (page.note or "")


async def test_a_browser_that_ended_on_another_site_is_a_redirect():
    blocked = _curl(_Reply(status=403, body=_CLOUDFLARE_HTML))
    browser = _FakeBrowser(_browser_page(final_url="https://elsewhere.com/landing"))
    with patch.object(fp, "AsyncSession", blocked):
        page = await fetch_url("https://a.com/", browser, redirects="same_site")

    assert page.classification is Classification.REDIRECT
    assert page.redirect_to == "https://elsewhere.com/landing"
    assert not page.text


async def test_any_redirect_is_followed_by_hand_across_sites():
    reply = _Reply(redirects={"https://bit.ly/x": (302, "https://shop.com/item")})
    with patch.object(fp, "AsyncSession", _curl(reply)):
        page = await fetch_url("https://bit.ly/x", None)

    assert page.classification is Classification.OK
    assert page.final_url == "https://shop.com/item"


@pytest.mark.parametrize("redirects", ["any", "same_site"])
async def test_a_redirect_to_a_private_address_is_not_followed(redirects):
    reply = _Reply(redirects={"https://a.com/": (302, "http://169.254.169.254/latest/")})
    curl = _curl(reply)
    browser = _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url("https://a.com/", browser, redirects=redirects)

    assert page.classification is Classification.NETWORK_ERROR
    assert "private IP" in (page.note or "")
    assert all(url == "https://a.com/" for _, url, _ in curl.calls)
    assert browser.calls == []  # not retried in the browser


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("file:///proc/self/environ", "only http and https"),
        ("ftp://a.com/f", "only http and https"),
        ("https://user:pw@a.com/", "credentials"),
        ("http://localhost:8000/", "local addresses"),
        ("http://127.0.0.1/", "private IP"),
        ("http://10.1.2.3/", "private IP"),
        ("http://192.168.0.1/", "private IP"),
        ("http://169.254.169.254/", "private IP"),
        ("http://[::1]/", "private IP"),
        ("http://2130706433/", "private IP"),  # 127.0.0.1, as curl reads it
        ("http://0x7f.1/", "private IP"),
        ("https://example.com/", None),
        ("https://8.8.8.8/dns", None),
    ],
)
def test_url_problem(url, problem):
    found = url_problem(url)
    assert (found is None) if problem is None else (problem in (found or ""))


async def test_a_url_that_must_not_be_fetched_is_refused_without_a_request():
    curl = _curl(_Reply())
    browser = _FakeBrowser(_browser_page())
    with patch.object(fp, "AsyncSession", curl):
        page = await fetch_url("file:///app/data/settings.json", browser)

    assert page.classification is Classification.NETWORK_ERROR
    assert "NOT retrieved" in (page.note or "") and not page.text
    assert curl.calls == [] and browser.calls == []


_FULL_PAGE_HTML = """<html><head><title>Bank holidays</title>
<style>.x{color:red}</style><script>var tracking = 1;</script></head>
<body><nav><a href="/about">About</a></nav><h1>Holidays</h1>
<table><tr><th>Date</th><th>Holiday</th></tr><tr><td>1 Jan</td><td>New Year</td></tr></table>
<p>Call <code>len(x)</code>, see <a href="#cite-1">[1]</a>.</p>
<noscript>Enable JavaScript</noscript><iframe src="https://ads.example">ad</iframe>
<template><p>template text</p></template></body></html>"""


def test_full_markdown_keeps_the_whole_page_as_markdown():
    md = full_markdown(_FULL_PAGE_HTML, "https://www.gov.uk/bank-holidays")

    assert md.startswith("# Bank holidays\n\n")
    assert md.count("Bank holidays") == 1
    assert "[About](https://www.gov.uk/about)" in md  # navigation kept, links absolute
    assert "# Holidays" in md
    assert "| 1 Jan | New Year |" in md  # every table column
    assert "`len(x)`" in md
    assert "see [1]." in md  # an in-page anchor is just its text


def test_full_markdown_drops_scripts_styles_and_frames():
    md = full_markdown(_FULL_PAGE_HTML, "https://www.gov.uk/bank-holidays")

    for gone in ("tracking", "color:red", "Enable JavaScript", "ads.example", "template text"):
        assert gone not in md


def test_full_markdown_of_nothing_is_empty():
    assert full_markdown("", "https://a.com/") == ""
