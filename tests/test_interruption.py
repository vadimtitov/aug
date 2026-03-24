"""Unit tests for mid-run injection and stop behaviour."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.base import _frame_injection
from aug.core.prompts import MID_RUN_INJECTION_PREFIX
from aug.core.run import AgentRun, RunRegistry

# ---------------------------------------------------------------------------
# AgentRun
# ---------------------------------------------------------------------------


def test_inject_message_str_queues_for_agent() -> None:
    run = AgentRun()
    run.inject_message("hello")
    assert run.pending_agent_injection.get_nowait() == "hello"


def test_inject_message_multimodal_queues_for_agent() -> None:
    run = AgentRun()
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    run.inject_message(content)
    assert run.pending_agent_injection.get_nowait() == content


def test_request_stop_sets_event() -> None:
    run = AgentRun()
    run.request_stop()
    assert run.user_requested_stop.is_set()


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------


def test_registry_same_thread_returns_same_lock() -> None:
    registry = RunRegistry()
    assert registry.thread_lock("t1") is registry.thread_lock("t1")


def test_registry_different_threads_return_different_locks() -> None:
    registry = RunRegistry()
    assert registry.thread_lock("t1") is not registry.thread_lock("t2")


def test_registry_clear_removes_runs_and_locks() -> None:
    registry = RunRegistry()
    registry.set("t1", AgentRun())
    registry.thread_lock("t1")
    registry.clear()
    assert registry.get("t1") is None
    assert registry.thread_lock("t1") is not None  # recreated fresh


# ---------------------------------------------------------------------------
# _frame_injection
# ---------------------------------------------------------------------------


def test_frame_injection_prepends_prefix_to_str() -> None:
    result = _frame_injection("do the thing")
    assert isinstance(result, str)
    assert result.startswith(MID_RUN_INJECTION_PREFIX)
    assert result.endswith("do the thing")


def test_frame_injection_prepends_text_block_to_multimodal() -> None:
    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    result = _frame_injection([image_block])
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": MID_RUN_INJECTION_PREFIX}
    assert result[1] == image_block


# ---------------------------------------------------------------------------
# Browser tool — stop via user_requested_stop event
# ---------------------------------------------------------------------------


@pytest.fixture()
def browser_deps():
    mock_thought = MagicMock()
    mock_thought.memory = "Navigated to example.com, found the widget listing."
    mock_thought.next_goal = "Click the first result."

    mock_history = MagicMock()
    mock_history.is_done.return_value = False  # default: run did not finish naturally
    mock_history.final_result.return_value = "Done."
    mock_history.screenshots.return_value = []
    mock_history.number_of_steps.return_value = 3
    mock_history.model_thoughts.return_value = [mock_thought]
    mock_history.urls.return_value = ["https://example.com/page"]
    mock_history.extracted_content.return_value = []

    mock_agent = MagicMock()
    mock_browser_instance = MagicMock()
    mock_browser_instance.stop = AsyncMock()
    mock_settings = MagicMock()
    mock_settings.BROWSER_CDP_URL = "http://chromium:9222"

    captured: dict = {}

    def capture_agent(**kwargs):
        captured["step_callback"] = kwargs.get("register_new_step_callback")
        mock_agent.history = mock_history
        return mock_agent

    mock_state = MagicMock()
    mock_state.url = "https://example.com/page"
    mock_output = MagicMock()
    mock_output.next_goal = "find the thing"

    async def run_one_step():
        if captured.get("step_callback"):
            await captured["step_callback"](mock_state, mock_output, 1)
        return mock_history

    mock_agent.run = AsyncMock(side_effect=run_one_step)

    with (
        patch("aug.core.tools.browser.get_settings", return_value=mock_settings),
        patch("aug.core.tools.browser.Agent", side_effect=capture_agent),
        patch("aug.core.tools.browser.Browser", return_value=mock_browser_instance),
        patch("aug.core.tools.browser._llm", return_value=MagicMock()),
        patch("aug.core.tools.browser.send_tool_progress_update", new=AsyncMock()),
        patch("aug.core.tools.browser.socket.gethostbyname", return_value="1.2.3.4"),
        patch("aug.core.tools.browser.Path.iterdir", return_value=iter([])),
        patch("aug.core.tools.browser.Path.exists", return_value=True),
    ):
        yield {"agent": mock_agent, "history": mock_history}


def _invoke_with_run(task: str, run: AgentRun):
    from aug.core.run import AGENT_RUN_CONFIG_KEY
    from aug.core.tools.browser import browser

    return browser.ainvoke(
        {"type": "tool_call", "id": "test-id", "name": "browser", "args": {"task": task}},
        config={"configurable": {AGENT_RUN_CONFIG_KEY: run}},
    )


async def test_browser_stops_when_stop_requested(browser_deps) -> None:
    run = AgentRun()
    run.request_stop()
    result = await _invoke_with_run("browse something", run)
    content = result.content
    assert "stopped after 3 steps" in content
    assert "Navigated to example.com" in content  # memory
    assert "Click the first result" in content  # next_goal
    assert "https://example.com/page" in content  # last url


async def test_browser_injects_new_user_input_into_agent(browser_deps) -> None:
    run = AgentRun()
    run.inject_message("buy apples instead")
    mock_agent = browser_deps["agent"]
    mock_agent.add_new_task = MagicMock()
    await _invoke_with_run("browse something", run)
    # message was consumed from queue and injected into the browser agent
    assert run.pending_agent_injection.qsize() == 0
    mock_agent.add_new_task.assert_called_once()
    injected = mock_agent.add_new_task.call_args[0][0]
    assert "buy apples instead" in injected


async def test_browser_completes_normally_without_stop(browser_deps) -> None:
    browser_deps["history"].is_done.return_value = True
    run = AgentRun()
    result = await _invoke_with_run("browse something", run)
    assert "stopped" not in result.content.lower()
    assert "interrupted" not in result.content.lower()
