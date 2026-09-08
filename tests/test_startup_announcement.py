"""Tests for the startup announcement.

Two layers: ``dispatch.broadcast`` fans a service-level message out over the
interface registry, and each interface answers ``announcement_threads`` for
itself.  The last test boots the real app to check the two are wired together.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.core.dispatch import broadcast


def _iface(threads: list[str]) -> MagicMock:
    iface = MagicMock()
    iface.announcement_threads = AsyncMock(return_value=threads)
    iface.send_proactive = AsyncMock()
    return iface


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broadcast_delivers_to_every_thread_of_every_interface():
    telegram = _iface(["tg-1-0", "tg-2-0"])
    other = _iface(["x-1"])
    app = SimpleNamespace(state=SimpleNamespace(interfaces={"telegram": telegram, "other": other}))

    assert await broadcast(app, "hello") == 3
    telegram.send_proactive.assert_any_await("tg-1-0", "hello")
    telegram.send_proactive.assert_any_await("tg-2-0", "hello")
    other.send_proactive.assert_awaited_once_with("x-1", "hello")


@pytest.mark.asyncio
async def test_broadcast_skips_interfaces_with_no_targets():
    rest = _iface([])
    app = SimpleNamespace(state=SimpleNamespace(interfaces={"rest_api": rest}))

    assert await broadcast(app, "hello") == 0
    rest.send_proactive.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_continues_past_a_failing_thread():
    telegram = _iface(["bad", "good"])
    telegram.send_proactive = AsyncMock(side_effect=[RuntimeError("Chat not found"), None])
    app = SimpleNamespace(state=SimpleNamespace(interfaces={"telegram": telegram}))

    assert await broadcast(app, "hello") == 1


@pytest.mark.asyncio
async def test_broadcast_continues_past_a_failing_interface():
    broken = _iface([])
    broken.announcement_threads = AsyncMock(side_effect=RuntimeError("boom"))
    working = _iface(["x-1"])
    app = SimpleNamespace(state=SimpleNamespace(interfaces={"broken": broken, "working": working}))

    assert await broadcast(app, "hello") == 1


@pytest.mark.asyncio
async def test_broadcast_with_no_interfaces_registered():
    app = SimpleNamespace(state=SimpleNamespace())
    assert await broadcast(app, "hello") == 0


# ---------------------------------------------------------------------------
# Interface targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rest_has_no_announcement_threads():
    from aug.api.interfaces.rest import RestApiInterface

    assert await RestApiInterface(checkpointer=MagicMock()).announcement_threads() == []


@pytest.mark.asyncio
async def test_telegram_announces_to_allowed_users_dms():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    iface = TelegramInterface(checkpointer=MagicMock())
    settings = MagicMock(allowed_chat_ids={222, 111, -1003820312204})

    with (
        patch("aug.api.interfaces.telegram.interface.get_settings", return_value=settings),
        patch(
            "aug.api.interfaces.telegram.utils.load_state",
            return_value=MagicMock(telegram=MagicMock(chats={})),
        ),
    ):
        threads = await iface.announcement_threads()

    # Groups (negative IDs) are excluded — announcements go to people, not rooms.
    assert threads == ["tg-111-0", "tg-222-0"]


@pytest.mark.asyncio
async def test_telegram_falls_back_to_known_dm_conversations():
    from aug.api.interfaces.telegram.interface import TelegramInterface
    from aug.utils.file_settings import AppSettings, ConversationSettings

    iface = TelegramInterface(checkpointer=MagicMock())
    settings = MagicMock(allowed_chat_ids=set())
    file_settings = AppSettings(
        conversations={
            "tg-555": ConversationSettings(),
            "tg--100123": ConversationSettings(),  # group — excluded
            "tg--100123-topic-7": ConversationSettings(),  # forum topic — excluded
            "rest-my-thread": ConversationSettings(),  # other interface — excluded
        }
    )

    with (
        patch("aug.api.interfaces.telegram.interface.get_settings", return_value=settings),
        patch("aug.api.interfaces.telegram.interface.load_settings", return_value=file_settings),
        patch(
            "aug.api.interfaces.telegram.utils.load_state",
            return_value=MagicMock(telegram=MagicMock(chats={})),
        ),
    ):
        threads = await iface.announcement_threads()

    assert threads == ["tg-555-0"]


@pytest.mark.asyncio
async def test_telegram_has_no_targets_when_nobody_is_known():
    from aug.api.interfaces.telegram.interface import TelegramInterface
    from aug.utils.file_settings import AppSettings

    iface = TelegramInterface(checkpointer=MagicMock())
    settings = MagicMock(allowed_chat_ids=set())

    with (
        patch("aug.api.interfaces.telegram.interface.get_settings", return_value=settings),
        patch("aug.api.interfaces.telegram.interface.load_settings", return_value=AppSettings()),
    ):
        assert await iface.announcement_threads() == []


# ---------------------------------------------------------------------------
# Wiring — the real FastAPI lifespan
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_ctx(value):
    yield value


def _boot(announce: bool) -> MagicMock:
    """Boot the real app with a fake interface registered, and return that interface.

    ``start_polling`` is replaced by a stub that registers the fake the way Telegram
    does, so this exercises the whole path — lifespan → background task → broadcast →
    interface — without a bot token or a network call.
    """
    from aug.api.interfaces.telegram.interface import TelegramInterface
    from aug.app import create_app

    iface = _iface(["fake-thread"])

    async def _register(self, app) -> None:
        app.state.interfaces["fake"] = iface

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    settings = MagicMock(
        STARTUP_ANNOUNCEMENT=announce,
        APP_VERSION="1.2.3",
        DEBUG=True,
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test",
        BROWSER_CDP_URL=None,
    )

    with (
        patch("aug.app.get_settings", return_value=settings),
        patch("aug.app.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(MagicMock())),
        patch("aug.app.init_memory_files"),
        patch("aug.app.start_consolidation_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.start_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.stop_scheduler", new=AsyncMock()),
        patch("aug.app.set_push_app"),
        patch("aug.app.set_pool"),
        patch("aug.app.serve_gateway", new=AsyncMock()),
        patch.object(TelegramInterface, "start_polling", new=_register),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        with TestClient(create_app()) as client:
            # The announcement runs as a background task; give it a turn to finish.
            client.portal.call(asyncio.sleep, 0.05)  # type: ignore[union-attr]
    return iface


def test_lifespan_announces_to_whatever_interface_registered_itself():
    iface = _boot(announce=True)

    iface.send_proactive.assert_awaited_once()
    thread_id, message = iface.send_proactive.await_args[0]
    assert thread_id == "fake-thread"
    assert message == "🟢 AUG 1.2.3 is up."


def test_lifespan_respects_the_opt_out():
    _boot(announce=False).send_proactive.assert_not_awaited()
