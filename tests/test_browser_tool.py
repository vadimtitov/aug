"""Unit tests for the browser tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.tools.browser import browser


def _invoke(task: str, **kwargs):
    """Invoke the browser tool as a ToolCall so LangChain wraps output in ToolMessage."""
    return browser.ainvoke(
        {
            "type": "tool_call",
            "id": "test-call-id",
            "name": "browser",
            "args": {"task": task, **kwargs},
        }
    )


@pytest.fixture()
def mock_browser_deps():
    """Patch browser-use classes and LLM."""
    # Minimal 1x1 PNG as base64 — matches what browser-use screenshots() returns
    _PNG_1PX = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()

    mock_history = MagicMock()
    mock_history.final_result.return_value = "Order placed."
    mock_history.screenshots.return_value = [_PNG_1PX]

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_history)

    mock_browser_instance = MagicMock()
    mock_browser_instance.stop = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.BROWSER_CDP_URL = "http://chromium:9222"

    with (
        patch("aug.core.tools.browser.get_settings", return_value=mock_settings),
        patch("aug.core.tools.browser.Agent", return_value=mock_agent) as mock_agent_cls,
        patch(
            "aug.core.tools.browser.Browser", return_value=mock_browser_instance
        ) as mock_browser_cls,
        patch("aug.core.tools.browser._llm", return_value=MagicMock()),
        patch("aug.core.tools.browser._build_tools", return_value=MagicMock()),
        patch("aug.core.tools.browser.socket.gethostbyname", return_value="1.2.3.4"),
        patch("aug.core.tools.browser.Path.iterdir", return_value=iter([])),
        patch("aug.core.tools.browser.Path.exists", return_value=True),
    ):
        yield {
            "agent_cls": mock_agent_cls,
            "browser_cls": mock_browser_cls,
            "agent": mock_agent,
            "browser": mock_browser_instance,
            "history": mock_history,
        }


async def test_browser_not_configured() -> None:
    mock_settings = MagicMock()
    mock_settings.BROWSER_CDP_URL = None
    with patch("aug.core.tools.browser.get_settings", return_value=mock_settings):
        result = await _invoke("order pizza")
    assert "not available" in result.content


async def test_browser_returns_final_result(mock_browser_deps) -> None:
    from aug.core.tools.output import ImageAttachment, ToolOutput

    result = await _invoke("check the weather")
    assert isinstance(result.artifact, ToolOutput)
    assert str(result.artifact) == "Order placed."
    assert len(result.artifact.attachments) == 1
    assert isinstance(result.artifact.attachments[0], ImageAttachment)


async def test_browser_passes_task_to_agent(mock_browser_deps) -> None:
    await _invoke("find the best pizza in London")
    kwargs = mock_browser_deps["agent_cls"].call_args.kwargs
    assert kwargs["task"] == "find the best pizza in London"


async def test_browser_connects_to_configured_url(mock_browser_deps) -> None:
    await _invoke("do something")
    call_kwargs = mock_browser_deps["browser_cls"].call_args.kwargs
    assert call_kwargs["cdp_url"] == "http://1.2.3.4:9222"


async def test_browser_stops_on_success(mock_browser_deps) -> None:
    await _invoke("search something")
    mock_browser_deps["browser"].stop.assert_awaited_once()


async def test_browser_stops_on_failure(mock_browser_deps) -> None:
    mock_browser_deps["agent"].run = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await _invoke("do something")
    assert "failed" in result.content.lower()
    mock_browser_deps["browser"].stop.assert_awaited_once()


async def test_browser_fallback_when_no_final_result(mock_browser_deps) -> None:
    mock_browser_deps["history"].final_result.return_value = None
    result = await _invoke("click around")
    content = result.content.lower()
    assert "not complete" in content or "failed" in content or "stopped" in content


# --- captcha action -------------------------------------------------------


def _session(screenshot=b"PNG", *, evaluate_value=None):
    """Mock BrowserSession. evaluate_value is what page JS (Runtime.evaluate) returns."""
    session = MagicMock()
    session.take_screenshot = AsyncMock(return_value=screenshot)
    cdp = MagicMock()
    cdp.session_id = "sess-1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": evaluate_value}}
    )
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    return session


def _vision_llm(content: str):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return llm


# A 1x1 PNG data URL, as returned by canvas.toDataURL in the page.
_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_build_tools_registers_captcha_action_and_keeps_defaults() -> None:
    from aug.core.tools.browser import _build_tools

    tools = _build_tools()
    names = list(tools.registry.registry.actions.keys())
    assert "ask_human_to_solve_captcha" in names
    assert "click" in names  # default actions preserved
    # No arguments — the human reads the whole screen, not a specified element.
    fields = tools.registry.registry.actions["ask_human_to_solve_captcha"].param_model.model_fields
    assert list(fields.keys()) == []


async def test_solve_captcha_prefers_raw_image_over_screenshot() -> None:
    from aug.core.tools.browser import _solve_captcha

    session = _session(evaluate_value=_PNG_DATA_URL)
    result = await _solve_captcha(_vision_llm("7XK9P"), session)
    assert "7XK9P" in result.extracted_content
    # Raw image was used; the viewport screenshot fallback was NOT taken.
    session.take_screenshot.assert_not_awaited()


async def test_solve_captcha_falls_back_to_screenshot_when_no_captcha_img() -> None:
    from aug.core.tools.browser import _solve_captcha

    session = _session(evaluate_value=None)  # JS found no captcha <img>
    await _solve_captcha(_vision_llm("7XK9P"), session)
    session.take_screenshot.assert_awaited_once_with()  # full viewport fallback


async def test_grab_captcha_image_decodes_data_url() -> None:
    import base64

    from aug.core.tools.browser import _grab_captcha_image

    session = _session(evaluate_value=_PNG_DATA_URL)
    data = await _grab_captcha_image(session)
    assert data == base64.b64decode(_PNG_DATA_URL.split(",", 1)[1])


async def test_grab_captcha_image_returns_none_when_no_image() -> None:
    from aug.core.tools.browser import _grab_captcha_image

    assert await _grab_captcha_image(_session(evaluate_value=None)) is None


async def test_transcribe_captcha_sends_image_and_strips() -> None:
    from aug.core.tools.browser import _transcribe_captcha

    llm = _vision_llm("  7XK9P\n")
    result = await _transcribe_captcha(llm, b"\x89PNG")
    assert result == "7XK9P"
    # The vision model received an image_url content block.
    messages = llm.ainvoke.call_args.args[0]
    human = messages[-1]
    assert human.content[0]["type"] == "image_url"
    assert human.content[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_solve_captcha_returns_human_voiced_solution() -> None:
    from aug.core.tools.browser import _solve_captcha

    result = await _solve_captcha(_vision_llm("7XK9P"), _session())
    assert "7XK9P" in result.extracted_content
    assert "human" in result.extracted_content.lower()


async def test_solve_captcha_unreadable_does_not_invent_value() -> None:
    from aug.core.tools.browser import _solve_captcha

    result = await _solve_captcha(_vision_llm("UNREADABLE"), _session())
    assert "could not read" in result.extracted_content.lower()
    assert "do not guess" in result.extracted_content.lower()


async def test_solve_captcha_handles_capture_failure() -> None:
    from aug.core.tools.browser import _solve_captcha

    session = MagicMock()
    session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("CDP gone"))
    result = await _solve_captcha(_vision_llm("x"), session)
    assert "could not check" in result.extracted_content.lower()
    assert "CDP gone" in result.extracted_content
