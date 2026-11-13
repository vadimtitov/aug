"""Tests for BaseInterface debounce — rapid messages merged into one run."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aug.api.interfaces.base import IncomingMessage, TextContent, _merge_messages
from aug.core.harness import Message, Sender, TextPart
from aug.core.run import RunRegistry
from tests.harness_fakes import ScriptedHarness

_SENDER = Sender(interface="telegram", user_id="1")


def _msg(*texts: str) -> Message:
    return Message(parts=[TextPart(text=t) for t in texts], sender=_SENDER)


async def _as_message(incoming: IncomingMessage) -> Message:
    return _msg(*(p.text for p in incoming.parts))


# ---------------------------------------------------------------------------
# _merge_messages
# ---------------------------------------------------------------------------


def test_merge_single_passthrough() -> None:
    message = _msg("hello")
    assert _merge_messages([message]) is message


def test_merge_concatenates_parts_in_order() -> None:
    merged = _merge_messages([_msg("hello"), _msg("world", "again")])
    assert [p.text for p in merged.parts] == ["hello", "world", "again"]


def test_merge_keeps_the_first_sender() -> None:
    other = Message(parts=[TextPart(text="b")], sender=Sender(interface="telegram", user_id="2"))
    assert _merge_messages([_msg("a"), other]).sender == _SENDER


# ---------------------------------------------------------------------------
# Debounce integration — BaseInterface.run()
# ---------------------------------------------------------------------------


def _make_incoming(text: str, context_id: str = "tg-1-0") -> IncomingMessage:
    return IncomingMessage(
        parts=[TextContent(text=text)],
        interface="telegram",
        sender_id="1",
        context_id=context_id,
        agent_version="default",
    )


@pytest.fixture()
def harness():
    harness = ScriptedHarness()
    with patch("aug.api.interfaces.base.get_harness", return_value=harness):
        yield harness


@pytest.fixture()
def isolated_registry():
    """Each test gets a clean RunRegistry so runs don't bleed across tests."""
    registry = RunRegistry()
    with patch("aug.api.interfaces.base.run_registry", registry):
        yield registry


@pytest.fixture()
def interface(isolated_registry, harness):
    from aug.api.interfaces.base import BaseInterface

    class _TestInterface(BaseInterface[str]):
        _debounce_window = 0.05  # 50ms — fast enough for tests

        async def receive_message(self, context: str) -> IncomingMessage:
            return _make_incoming(context)

        async def send_notification(self, target_id: str, text: str) -> None:
            pass

        async def resolve_context(self, context_id, *, topic_name=None, chat_id=None) -> str:
            return context_id

        def conversation_id(self, context_id: str) -> str:
            return context_id

        async def send_proactive(self, context_id: str, text: str) -> None:
            pass

        async def announcement_threads(self) -> list[str]:
            return []

        async def send_proactive_stream(self, context_id, stream) -> None:
            async for _ in stream:
                pass

    iface = _TestInterface()
    iface._converse = AsyncMock()
    return iface


async def test_single_message_starts_one_run(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", _as_message):
        await interface.run("hello")
        await asyncio.sleep(0.1)

    interface._converse.assert_awaited_once()
    _incoming, merged, _context = interface._converse.call_args[0]
    assert merged == _msg("hello")


async def test_rapid_messages_merged_into_one_run(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", _as_message):
        await asyncio.gather(
            interface.run("msg1"),
            interface.run("msg2"),
            interface.run("msg3"),
        )
        await asyncio.sleep(0.1)

    interface._converse.assert_awaited_once()
    merged = interface._converse.call_args[0][1]
    assert {p.text for p in merged.parts} == {"msg1", "msg2", "msg3"}


async def test_message_during_active_run_is_injected(interface, harness) -> None:
    harness.accept_inject = True

    with patch("aug.api.interfaces.base._preprocess", _as_message):
        await interface.run("inject me")

    interface._converse.assert_not_awaited()
    assert harness.injected == [(_msg("inject me"), "tg-1-0")]


async def test_debounce_task_cleaned_up_after_fire(interface) -> None:
    with patch("aug.api.interfaces.base._preprocess", _as_message):
        await interface.run("hello")
        await asyncio.sleep(0.1)

    assert "tg-1-0" not in interface._debounce_tasks
    assert "tg-1-0" not in interface._debounce_buf
