"""Tests for BaseInterface debounce — rapid messages merged into one run."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.api.interfaces.base import IncomingMessage, TextContent, _merge_contents
from aug.core.run import AgentRun, RunRegistry

# ---------------------------------------------------------------------------
# _merge_contents
# ---------------------------------------------------------------------------


def test_merge_single_passthrough() -> None:
    assert _merge_contents(["hello"]) == "hello"


def test_merge_two_strings() -> None:
    result = _merge_contents(["hello", "world"])
    assert result == "hello\n\nworld"


def test_merge_multimodal_blocks() -> None:
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    next_block = {"type": "text", "text": "next"}
    result = _merge_contents([[img, {"type": "text", "text": "caption"}], [next_block]])
    assert isinstance(result, list)
    assert result[0] == img
    assert result[1] == {"type": "text", "text": "caption"}
    assert result[2] == {"type": "text", "text": "next"}


def test_merge_mixed_str_and_blocks() -> None:
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    result = _merge_contents(["text first", [img]])
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "text first"}
    assert result[1] == img


def test_merge_skips_empty_strings() -> None:
    result = _merge_contents(["", "hello", ""])
    assert result == "hello"


# ---------------------------------------------------------------------------
# Debounce integration — BaseInterface.run()
# ---------------------------------------------------------------------------


def _make_incoming(text: str, thread_id: str = "tg-1-0") -> IncomingMessage:
    return IncomingMessage(
        parts=[TextContent(text=text)],
        interface="telegram",
        sender_id="1",
        thread_id=thread_id,
        agent_version="default",
    )


@pytest.fixture()
def isolated_registry():
    """Each test gets a clean RunRegistry so runs don't bleed across tests."""
    registry = RunRegistry()
    with patch("aug.api.interfaces.base.run_registry", registry):
        yield registry


@pytest.fixture()
def interface(isolated_registry):
    from aug.api.interfaces.base import BaseInterface
    from aug.core.tools.approval import ApprovalRequest

    class _TestInterface(BaseInterface[str]):
        _debounce_window = 0.05  # 50ms — fast enough for tests

        async def receive_message(self, context: str) -> IncomingMessage:
            return _make_incoming(context)

        async def send_notification(self, target_id: str, text: str) -> None:
            pass

        async def request_approval(self, request: ApprovalRequest, context: str) -> None:
            pass

    iface = _TestInterface(checkpointer=MagicMock())
    iface._execute_run = AsyncMock()
    return iface


async def test_single_message_starts_one_run(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", AsyncMock(side_effect=lambda p: p[0].text)):
        await interface.run("hello")
        await asyncio.sleep(0.1)

    interface._execute_run.assert_awaited_once()
    _, merged, _ = interface._execute_run.call_args[0][1:4]  # (run, incoming, content, context)
    assert merged == "hello"


async def test_rapid_messages_merged_into_one_run(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", AsyncMock(side_effect=lambda p: p[0].text)):
        await asyncio.gather(
            interface.run("msg1"),
            interface.run("msg2"),
            interface.run("msg3"),
        )
        await asyncio.sleep(0.1)

    interface._execute_run.assert_awaited_once()
    content = interface._execute_run.call_args[0][2]
    assert "msg1" in content
    assert "msg2" in content
    assert "msg3" in content


async def test_message_during_active_run_is_injected(interface, isolated_registry) -> None:
    active_run = AgentRun()
    active_run.active = True
    isolated_registry.set("tg-1-0", active_run)

    with patch("aug.api.interfaces.base._preprocess", AsyncMock(side_effect=lambda p: p[0].text)):
        await interface.run("inject me")

    interface._execute_run.assert_not_awaited()
    assert active_run.pending_agent_injection.get_nowait() == "inject me"


async def test_debounce_task_cleaned_up_after_fire(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", AsyncMock(side_effect=lambda p: p[0].text)):
        await interface.run("hello")
        await asyncio.sleep(0.1)

    assert "tg-1-0" not in interface._debounce_tasks
    assert "tg-1-0" not in interface._debounce_buf
