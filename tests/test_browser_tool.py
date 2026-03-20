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
