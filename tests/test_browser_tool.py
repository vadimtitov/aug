"""Unit tests for the browser tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.tools.browser import browser


@pytest.fixture()
def mock_browser_deps():
    """Patch browser-use classes and LLM."""
    mock_history = MagicMock()
    mock_history.final_result.return_value = "Order placed."

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
        result = await browser.ainvoke({"task": "order pizza"})
    assert "not available" in result


async def test_browser_returns_final_result(mock_browser_deps) -> None:
    result = await browser.ainvoke({"task": "check the weather"})
    assert result == "Order placed."


async def test_browser_passes_task_to_agent(mock_browser_deps) -> None:
    await browser.ainvoke({"task": "find the best pizza in London"})
    kwargs = mock_browser_deps["agent_cls"].call_args.kwargs
    assert kwargs["task"] == "find the best pizza in London"


async def test_browser_connects_to_configured_url(mock_browser_deps) -> None:
    await browser.ainvoke({"task": "do something"})
    mock_browser_deps["browser_cls"].assert_called_once_with(cdp_url="http://1.2.3.4:9222")


async def test_browser_stops_on_success(mock_browser_deps) -> None:
    await browser.ainvoke({"task": "search something"})
    mock_browser_deps["browser"].stop.assert_awaited_once()


async def test_browser_stops_on_failure(mock_browser_deps) -> None:
    mock_browser_deps["agent"].run = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await browser.ainvoke({"task": "do something"})
    assert "failed" in result.lower()
    mock_browser_deps["browser"].stop.assert_awaited_once()


async def test_browser_fallback_when_no_final_result(mock_browser_deps) -> None:
    mock_browser_deps["history"].final_result.return_value = None
    result = await browser.ainvoke({"task": "click around"})
    assert (
        "failed" in result.lower()
        or "not completing" in result.lower()
        or "stopped" in result.lower()
    )
