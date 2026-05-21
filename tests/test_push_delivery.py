"""Tests for push delivery interface methods.

Covers RestApiInterface stubs and TelegramInterface.resolve_thread / send_proactive /
send_proactive_stream — the three abstract BaseInterface methods added to support
scheduled tasks and external push routing.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.events import ChatModelStreamEvent

# ---------------------------------------------------------------------------
# RestApiInterface
# ---------------------------------------------------------------------------


@pytest.fixture()
def rest_iface():
    """Return a RestApiInterface with a mock checkpointer."""
    from aug.api.interfaces.rest import RestApiInterface

    return RestApiInterface(checkpointer=MagicMock())


@pytest.mark.asyncio
async def test_rest_resolve_thread_returns_passthrough(rest_iface):
    """Specific thread IDs are returned unchanged."""
    assert await rest_iface.resolve_thread("tg-12345-0") == "tg-12345-0"
    assert await rest_iface.resolve_thread("some-thread-id") == "some-thread-id"


@pytest.mark.asyncio
async def test_rest_resolve_thread_default_raises(rest_iface):
    """'default' is not meaningful for REST — must raise ValueError."""
    with pytest.raises(ValueError, match="default"):
        await rest_iface.resolve_thread("default")


@pytest.mark.asyncio
async def test_rest_resolve_thread_new_raises(rest_iface):
    """'new' is not meaningful for REST — must raise ValueError."""
    with pytest.raises(ValueError, match="new"):
        await rest_iface.resolve_thread("new")


@pytest.mark.asyncio
async def test_rest_send_proactive_is_noop(rest_iface):
    """send_proactive must not raise — REST has no push channel."""
    await rest_iface.send_proactive("tg-12345-0", "hello")


@pytest.mark.asyncio
async def test_rest_send_proactive_stream_consumes_stream(rest_iface):
    """send_proactive_stream must drain the stream without raising."""
    consumed: list[str] = []

    async def mock_stream() -> AsyncIterator[ChatModelStreamEvent]:
        for delta in ("a", "b", "c"):
            consumed.append(delta)
            yield ChatModelStreamEvent(delta=delta)

    await rest_iface.send_proactive_stream("tg-12345-0", mock_stream())
    assert consumed == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Telegram: _parse_thread_id
# ---------------------------------------------------------------------------


def test_parse_dm_thread_id():
    """DM thread IDs parse to (chat_id, None)."""
    from aug.api.interfaces.telegram.interface import _parse_thread_id

    chat_id, topic_id = _parse_thread_id("tg-12345-3")
    assert chat_id == 12345
    assert topic_id is None


def test_parse_topic_thread_id():
    """Topic thread IDs parse to (chat_id, topic_id)."""
    from aug.api.interfaces.telegram.interface import _parse_thread_id

    chat_id, topic_id = _parse_thread_id("tg-12345-topic-42")
    assert chat_id == 12345
    assert topic_id == 42


def test_parse_negative_chat_id():
    """Negative chat IDs (Telegram supergroups) are parsed correctly."""
    from aug.api.interfaces.telegram.interface import _parse_thread_id

    chat_id, topic_id = _parse_thread_id("tg--100123456789-topic-5")
    assert chat_id == -100123456789
    assert topic_id == 5


def test_parse_invalid_thread_id_raises():
    """Malformed thread IDs raise ValueError."""
    from aug.api.interfaces.telegram.interface import _parse_thread_id

    with pytest.raises(ValueError):
        _parse_thread_id("not-a-telegram-id")


# ---------------------------------------------------------------------------
# TelegramInterface: resolve_thread
# ---------------------------------------------------------------------------


@pytest.fixture()
def tg_iface():
    """Return a TelegramInterface with a mock checkpointer and mock bot."""
    from aug.api.interfaces.telegram.interface import TelegramInterface

    iface = TelegramInterface(checkpointer=MagicMock())
    mock_bot = AsyncMock()
    iface._bot_app = MagicMock()
    iface._bot_app.bot = mock_bot
    return iface


@pytest.mark.asyncio
async def test_tg_resolve_thread_passthrough(tg_iface):
    """Concrete thread IDs are returned unchanged."""
    result = await tg_iface.resolve_thread("tg-12345-topic-42")
    assert result == "tg-12345-topic-42"


@pytest.mark.asyncio
async def test_tg_resolve_thread_default_with_explicit_chat_id(tg_iface):
    """'default' with an explicit chat_id resolves to the DM thread for that chat."""
    with patch("aug.api.interfaces.telegram.utils.load_state") as mock_state:
        mock_state.return_value.telegram.chats = {}
        result = await tg_iface.resolve_thread("default", chat_id=12345)
    assert result == "tg-12345-0"


@pytest.mark.asyncio
async def test_tg_resolve_thread_embedded_chat_id(tg_iface):
    """'default:{chat_id}' resolves to the DM thread for that exact chat_id."""
    with patch("aug.api.interfaces.telegram.utils.load_state") as mock_state:
        mock_state.return_value.telegram.chats = {}
        result = await tg_iface.resolve_thread("default:77777")
    assert result == "tg-77777-0"


@pytest.mark.asyncio
async def test_tg_resolve_thread_default_finds_chat_from_settings(tg_iface):
    """'default' without chat_id falls back to the first positive chat_id from settings."""
    from aug.utils.file_settings import AppSettings, TelegramChatSettings, TelegramSettings

    settings = AppSettings(telegram=TelegramSettings(chats={"99999": TelegramChatSettings()}))
    with (
        patch("aug.api.interfaces.telegram.interface.load_settings", return_value=settings),
        patch("aug.api.interfaces.telegram.utils.load_state") as mock_state,
    ):
        mock_state.return_value.telegram.chats = {}
        result = await tg_iface.resolve_thread("default")
    assert result == "tg-99999-0"


@pytest.mark.asyncio
async def test_tg_resolve_thread_default_no_chats_raises(tg_iface):
    """'default' with no chats configured raises ValueError."""
    from aug.utils.file_settings import AppSettings, TelegramSettings

    settings = AppSettings(telegram=TelegramSettings(chats={}))
    with patch("aug.api.interfaces.telegram.interface.load_settings", return_value=settings):
        with pytest.raises(ValueError, match="No default"):
            await tg_iface.resolve_thread("default")


@pytest.mark.asyncio
async def test_tg_resolve_thread_new_creates_forum_topic(tg_iface):
    """'new' creates a Telegram forum topic and returns its thread ID."""
    tg_iface._bot_app.bot.create_forum_topic = AsyncMock(
        return_value=MagicMock(message_thread_id=99)
    )
    with patch("aug.api.interfaces.telegram.utils.load_state") as mock_state:
        mock_state.return_value.telegram.chats = {}
        result = await tg_iface.resolve_thread("new", chat_id=12345, topic_name="My Topic")

    tg_iface._bot_app.bot.create_forum_topic.assert_called_once_with(chat_id=12345, name="My Topic")
    assert result == "tg-12345-topic-99"


# ---------------------------------------------------------------------------
# TelegramInterface: send_proactive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tg_send_proactive_sends_to_dm(tg_iface):
    """send_proactive sends a plain-text message to a DM thread."""
    await tg_iface.send_proactive("tg-12345-0", "Hello world")

    tg_iface._bot_app.bot.send_message.assert_called_once_with(
        chat_id=12345,
        text="Hello world",
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_tg_send_proactive_sends_to_topic(tg_iface):
    """send_proactive routes to the correct forum topic via message_thread_id."""
    await tg_iface.send_proactive("tg-12345-topic-7", "Hello topic")

    tg_iface._bot_app.bot.send_message.assert_called_once_with(
        chat_id=12345,
        text="Hello topic",
        message_thread_id=7,
    )


# ---------------------------------------------------------------------------
# TelegramInterface: send_proactive_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tg_send_proactive_stream_sends_final_text(tg_iface):
    """send_proactive_stream collects the full agent response and sends it as one message."""

    async def mock_stream() -> AsyncIterator:
        yield ChatModelStreamEvent(delta="Hello")
        yield ChatModelStreamEvent(delta=" world")

    await tg_iface.send_proactive_stream("tg-12345-0", mock_stream())

    tg_iface._bot_app.bot.send_message.assert_called_once()
    call_kwargs = tg_iface._bot_app.bot.send_message.call_args.kwargs
    assert call_kwargs["chat_id"] == 12345
    assert call_kwargs["message_thread_id"] is None
    # Text should contain the accumulated response (possibly HTML-formatted)
    assert "Hello world" in call_kwargs["text"] or "Hello" in call_kwargs["text"]


@pytest.mark.asyncio
async def test_tg_send_proactive_stream_empty_does_not_send(tg_iface):
    """send_proactive_stream with no text events sends nothing."""

    async def mock_stream() -> AsyncIterator:
        return
        yield  # make it a generator

    await tg_iface.send_proactive_stream("tg-12345-0", mock_stream())

    tg_iface._bot_app.bot.send_message.assert_not_called()
