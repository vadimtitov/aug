"""web_fetch: WebFetch's contract (prompt in, a small model's answer out) on AUG's fetching.

Fetching is faked at fetch_url, the model at the ask callable / ToolContext.oneshot.
"""

from unittest.mock import AsyncMock, patch

import pytest

from aug.core.harness.tool_context import ToolContext
from aug.core.harness.types import Address, ModelTier, TurnContext
from aug.core.prompts import WEB_FETCH_INSTRUCTIONS
from aug.core.tools import web_fetch as wf
from aug.utils.fetching.fetcher import FetchedPage
from aug.utils.fetching.page_content import Classification


@pytest.fixture(autouse=True)
def _fresh_cache():
    wf.pages.clear()
    yield
    wf.pages.clear()


class _Model:
    """Stands in for the small model: records what it read, answers *answer*."""

    def __init__(self, answer: str = "The answer.", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.answer


def _page(text: str = "# Title\n\nBody text.", **kw) -> FetchedPage:
    base = dict(
        url="https://a.com/",
        tier="cheap",
        classification=Classification.OK,
        status=200,
        elapsed_s=0.1,
        text=text,
    )
    return FetchedPage(**{**base, **kw})


def _fetched(page: FetchedPage) -> AsyncMock:
    return AsyncMock(return_value=page)


async def test_the_model_reads_the_page_and_the_prompt_in_webfetchs_shape():
    model = _Model("Plans: Free, Pro.")
    with patch.object(wf, "fetch_url", _fetched(_page("Pricing: Free, Pro"))):
        out = await wf.read_page("https://a.com/", "List the plans", model)

    assert out == "Plans: Free, Pro."
    assert model.prompts == [
        "\nWeb page content:\n---\nPricing: Free, Pro\n---\n\n"
        f"List the plans\n\n{WEB_FETCH_INSTRUCTIONS}\n"
    ]


async def test_fetching_follows_only_same_site_redirects():
    fetch = _fetched(_page())
    with patch.object(wf, "fetch_url", fetch):
        await wf.read_page("https://a.com/", "q", _Model())

    assert fetch.call_args.kwargs == {"redirects": "same_site"}


async def test_a_long_page_is_read_up_to_the_limit_with_the_next_offset():
    model = _Model("Part one.")
    with patch.object(wf, "fetch_url", _fetched(_page("x" * 250))):
        out = await wf.read_page("https://a.com/", "q", model, max_chars=100)

    assert "x" * 100 + "\n\n[Content truncated due to length...]" in model.prompts[0]
    assert "x" * 101 not in model.prompts[0]
    assert out == (
        "Part one.\n[web_fetch note: the answer above covers characters 0 to 100 of this "
        "page's 250 characters. Next offset: 100.]"
    )


async def test_reading_on_from_an_offset_says_where_the_part_starts():
    model = _Model("Part two.")
    text = "a" * 100 + "b" * 100 + "c" * 50
    with patch.object(wf, "fetch_url", _fetched(_page(text))):
        out = await wf.read_page("https://a.com/", "q", model, offset=100, max_chars=100)

    assert model.prompts[0].startswith(
        "The content below is one part of a longer page: it starts 100 characters into "
        "the page's 250.\n\nWeb page content:\n---\n" + "b" * 100
    )
    assert "Next offset: 200." in out


async def test_the_last_part_has_no_note():
    with patch.object(wf, "fetch_url", _fetched(_page("a" * 150))):
        out = await wf.read_page("https://a.com/", "q", _Model("End."), offset=100)

    assert out == "End."


async def test_an_offset_past_the_end_says_so_without_a_model_call():
    model = _Model()
    with patch.object(wf, "fetch_url", _fetched(_page("a" * 50))):
        out = await wf.read_page("https://a.com/", "q", model, offset=50)

    assert out == "Nothing left to read from offset 50: this page's text is 50 characters long."
    assert model.prompts == []


async def test_raw_returns_the_page_text_itself():
    model = _Model()
    with patch.object(wf, "fetch_url", _fetched(_page("# Title\n\nBody."))):
        out = await wf.read_page("https://a.com/", "q", model, raw=True)

    assert out == "Fetched https://a.com/ (HTTP 200, 14 characters)\n\n# Title\n\nBody."
    assert model.prompts == []


async def test_raw_on_a_long_page_gives_the_next_offset():
    text = "a" * (wf.RAW_MAX_CHARS + 10)
    with patch.object(wf, "fetch_url", _fetched(_page(text))):
        out = await wf.read_page("https://a.com/", "q", _Model(), raw=True, offset=5)

    assert out.startswith(f"Fetched https://a.com/ (HTTP 200, {len(text)} characters, shown")
    assert f"Next offset: {wf.RAW_MAX_CHARS + 5}.]" in out


async def test_a_redirect_off_the_site_tells_the_agent_how_to_call_again():
    page = _page(
        "",
        url="https://bit.ly/x",
        final_url="https://bit.ly/x",
        classification=Classification.REDIRECT,
        status=301,
        redirect_to="https://shop.com/p",
    )
    model = _Model()
    with patch.object(wf, "fetch_url", _fetched(page)):
        out = await wf.read_page("https://bit.ly/x", "price?", model)

    assert out.startswith("REDIRECT DETECTED: The URL redirects to a location that was not")
    assert "Original URL: https://bit.ly/x\nRedirect URL: https://shop.com/p\nStatus: 301" in out
    assert 'url: "https://shop.com/p"' in out and 'prompt: "price?"' in out
    assert model.prompts == []


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (
            _page("", classification=Classification.BLOCKED, status=403, note="[FAILED — x]"),
            "NOT retrieved",
        ),
        (_page("Not found", classification=Classification.HTTP_ERROR, status=404), "HTTP 404"),
        (_page("", note="[binary content: application/pdf, 10 bytes; not extracted]"), "binary"),
    ],
)
async def test_a_page_that_could_not_be_read_fails_plainly_without_a_model_call(page, expected):
    model = _Model()
    with patch.object(wf, "fetch_url", _fetched(page)):
        out = await wf.read_page("https://a.com/", "q", model)

    assert out.startswith("Failed to fetch https://a.com/") and expected in out
    assert model.prompts == []


async def test_a_thin_page_is_read_with_its_note():
    page = _page("Little.", classification=Classification.THIN, note="[page has little text]")
    with patch.object(wf, "fetch_url", _fetched(page)):
        out = await wf.read_page("https://a.com/", "q", _Model("Hardly anything."))

    assert out == "[web_fetch note: [page has little text]]\nHardly anything."


async def test_a_model_that_fails_is_reported_as_not_read():
    with patch.object(wf, "fetch_url", _fetched(_page())):
        out = await wf.read_page("https://a.com/", "q", _Model(error=RuntimeError("quota")))

    assert "the model that reads the page failed: RuntimeError: quota" in out
    assert "NOT read" in out


async def test_an_empty_answer_is_reported():
    with patch.object(wf, "fetch_url", _fetched(_page())):
        out = await wf.read_page("https://a.com/", "q", _Model("  "))

    assert "gave no answer" in out


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("http://localhost:8000/", "no dot"),
        ("ftp://a.com/file", "only http and https"),
        ("https://user:pw@a.com/", "credentials"),
        ("https://a.com/" + "x" * 2000, "longer than 2000"),
    ],
)
async def test_urls_it_cannot_fetch_are_refused_before_fetching(url, problem):
    fetch = _fetched(_page())
    with patch.object(wf, "fetch_url", fetch):
        out = await wf.read_page(url, "q", _Model())

    assert out.startswith("Cannot fetch") and problem in out
    fetch.assert_not_awaited()


async def test_http_is_upgraded_to_https():
    fetch = _fetched(_page())
    with patch.object(wf, "fetch_url", fetch):
        await wf.read_page("http://a.com/x?y=1", "q", _Model())

    assert fetch.call_args.args[0] == "https://a.com/x?y=1"


async def test_a_page_is_fetched_once_for_several_questions():
    fetch = _fetched(_page("a" * 300))
    with patch.object(wf, "fetch_url", fetch):
        await wf.read_page("https://a.com/", "q1", _Model(), max_chars=100)
        await wf.read_page("https://a.com/", "q2", _Model(), offset=100, max_chars=100)

    assert fetch.await_count == 1


async def test_a_failed_fetch_is_not_kept():
    fetch = _fetched(_page("", classification=Classification.NETWORK_ERROR, status=0))
    with patch.object(wf, "fetch_url", fetch):
        await wf.read_page("https://a.com/", "q", _Model())
        await wf.read_page("https://a.com/", "q", _Model())

    assert fetch.await_count == 2


def test_the_cache_forgets_a_page_after_its_ttl():
    clock = [0.0]
    with patch.object(wf.time, "monotonic", lambda: clock[0]):
        cache = wf.PageCache(ttl_s=10)
        cache.put("https://a.com/", _page())
        clock[0] = 9
        assert cache.get("https://a.com/") is not None
        clock[0] = 11
        assert cache.get("https://a.com/") is None


async def test_the_tool_asks_the_harness_fast_model():
    oneshot = AsyncMock(return_value="From the fast model.")
    ctx = ToolContext(
        context_id="c",
        turn=TurnContext(Address(interface="telegram", conversation_id="1"), attended=True),
        _oneshot=oneshot,
    )
    with (
        patch.object(wf, "fetch_url", _fetched(_page())),
        patch.object(wf, "current_tool_context", return_value=ctx),
    ):
        out = await wf.web_fetch.ainvoke({"url": "https://a.com/", "prompt": "q"})

    assert out == "From the fast model."
    assert oneshot.call_args.args[1] is ModelTier.FAST


async def test_the_tool_outside_a_turn_reports_that_no_model_read_the_page():
    with (
        patch.object(wf, "fetch_url", _fetched(_page())),
        patch.object(wf, "current_tool_context", return_value=None),
    ):
        out = await wf.web_fetch.ainvoke({"url": "https://a.com/", "prompt": "q"})

    assert "NOT read" in out


async def test_a_tool_context_without_a_model_raises():
    ctx = ToolContext(
        context_id="c",
        turn=TurnContext(Address(interface="telegram", conversation_id="1"), attended=True),
    )
    with pytest.raises(RuntimeError, match="no model call"):
        await ctx.oneshot("hi")
