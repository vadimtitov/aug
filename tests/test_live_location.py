"""Tests for Telegram live location tracking (edited_message updates)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Location, Message, Update, User
from telegram.ext import filters

from aug.core.run import AgentRun, run_registry
from aug.utils.state import AppState, TelegramChatState

CHAT_ID = 123
USER = User(id=7, first_name="V", is_bot=False)
CHAT = Chat(id=CHAT_ID, type=Chat.PRIVATE)


@pytest.fixture(autouse=True)
def allow_all_chats():
    with patch("aug.api.interfaces.telegram.interface.is_allowed", return_value=True):
        yield


@pytest.fixture(autouse=True)
def clean_registry():
    run_registry.clear()
    yield
    run_registry.clear()


@pytest.fixture()
def telegram_interface():
    from aug.api.interfaces.telegram.interface import TelegramInterface

    return TelegramInterface(checkpointer=MagicMock())


def _message(**kwargs) -> Message:
    msg = Message(message_id=1, date=None, chat=CHAT, from_user=USER, **kwargs)
    msg.set_bot(MagicMock())
    return msg


def _edited_location(lat=1.5, lon=2.5, live_period=None, topic_id=None) -> Update:
    return Update(
        update_id=1,
        edited_message=_message(
            location=Location(latitude=lat, longitude=lon, live_period=live_period),
            message_thread_id=topic_id,
        ),
    )


def _state(**live_kwargs) -> AppState:
    state = AppState()
    state.telegram.chats[str(CHAT_ID)] = TelegramChatState(session=0)
    for key, value in live_kwargs.items():
        setattr(state.telegram.chats[str(CHAT_ID)].live_location, key, value)
    return state


async def _handle(interface, update, state):
    """Run _handle_live_location against *state*, returning (saved_states, run_mock)."""
    saved: list[AppState] = []
    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch.object(interface, "run", new=AsyncMock()) as run_mock,
    ):
        await interface._handle_live_location(update, MagicMock())
    return saved, run_mock


# ---------------------------------------------------------------------------
# State: backward compatibility and round-trip
# ---------------------------------------------------------------------------


def test_old_state_file_without_live_location_still_loads():
    from aug.utils.state import load_state

    raw = json.dumps({"telegram": {"chats": {"123": {"session": 42}}}})
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()

    assert s.telegram.chats["123"].session == 42
    assert s.telegram.chats["123"].live_location.latitude == 0.0
    assert s.telegram.chats["123"].live_location.throttle_seconds == 300


def test_live_location_defaults_are_not_shared_between_chats():
    state = AppState()
    state.telegram.chats["a"] = TelegramChatState()
    state.telegram.chats["b"] = TelegramChatState()
    state.telegram.chats["a"].live_location.latitude = 10.0

    assert state.telegram.chats["b"].live_location.latitude == 0.0


def test_save_round_trips_live_location():
    from aug.utils.state import save_state

    written: list[str] = []
    s = _state(latitude=1.5, longitude=2.5, updated_at=99.0, throttle_seconds=600)

    with patch("aug.utils.state.write_data_file", side_effect=lambda _f, d: written.append(d)):
        save_state(s)

    loaded = AppState.model_validate_json(written[0])
    assert loaded.telegram.chats["123"].live_location.latitude == 1.5
    assert loaded.telegram.chats["123"].live_location.throttle_seconds == 600


# ---------------------------------------------------------------------------
# Handler routing — the filters must be mutually exclusive
# ---------------------------------------------------------------------------


def _matching_handlers(bot_app, update):
    return [
        h.callback.__name__
        for group in bot_app.handlers.values()
        for h in group
        if h.check_update(update) not in (False, None)
    ]


@pytest.fixture()
def bot_app(telegram_interface):
    with patch("aug.api.interfaces.telegram.interface.get_settings") as settings:
        settings.return_value.TELEGRAM_BOT_TOKEN = "1:TESTTOKEN"
        return telegram_interface.build_bot()


def test_edited_location_routes_to_live_location_handler(bot_app):
    assert _matching_handlers(bot_app, _edited_location()) == ["_handle_live_location"]


def test_new_location_routes_to_handle_input(bot_app):
    update = Update(update_id=1, message=_message(location=Location(latitude=1.0, longitude=2.0)))
    assert _matching_handlers(bot_app, update) == ["_handle_input"]


def test_edited_text_message_never_reaches_live_location(bot_app):
    """An edited text message must not be treated as a location update."""
    update = Update(update_id=1, edited_message=_message(text="typo fixed"))
    assert "_handle_live_location" not in _matching_handlers(bot_app, update)


@pytest.mark.asyncio
async def test_edited_text_message_is_dropped(telegram_interface):
    """_handle_text matches edited text, but its update.message guard drops it."""
    update = Update(update_id=1, edited_message=_message(text="typo fixed"))
    with patch.object(telegram_interface, "run", new=AsyncMock()) as run_mock:
        await telegram_interface._handle_text(update, MagicMock())
    run_mock.assert_not_awaited()


def test_filters_are_mutually_exclusive():
    edited_only = filters.LOCATION & filters.UpdateType.EDITED_MESSAGE
    new_only = filters.LOCATION & ~filters.UpdateType.EDITED_MESSAGE
    edited = _edited_location()
    new = Update(update_id=1, message=_message(location=Location(latitude=1.0, longitude=2.0)))

    assert edited_only.check_update(edited) and not new_only.check_update(edited)
    assert new_only.check_update(new) and not edited_only.check_update(new)


# ---------------------------------------------------------------------------
# _handle_live_location behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinates_are_persisted(telegram_interface):
    state = _state(last_run_at=0.0)
    saved, _ = await _handle(telegram_interface, _edited_location(lat=51.5, lon=-0.12), state)

    live = saved[0].telegram.chats["123"].live_location
    assert (live.latitude, live.longitude) == (51.5, -0.12)
    assert live.updated_at > 0


@pytest.mark.asyncio
async def test_live_period_sets_expiry(telegram_interface):
    state = _state()
    saved, _ = await _handle(telegram_interface, _edited_location(live_period=3600), state)

    live = saved[0].telegram.chats["123"].live_location
    assert live.live_until == pytest.approx(live.updated_at + 3600)


@pytest.mark.asyncio
async def test_missing_live_period_keeps_previous_expiry(telegram_interface):
    state = _state(live_until=12345.0)
    saved, _ = await _handle(telegram_interface, _edited_location(), state)

    assert saved[0].telegram.chats["123"].live_location.live_until == 12345.0


@pytest.mark.asyncio
async def test_throttled_update_saves_but_does_not_run(telegram_interface):
    import time

    state = _state(last_run_at=time.time(), throttle_seconds=300)
    saved, run_mock = await _handle(telegram_interface, _edited_location(lat=9.0), state)

    assert saved[0].telegram.chats["123"].live_location.latitude == 9.0
    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_elapsed_throttle_starts_a_run_and_stamps_last_run_at(telegram_interface):
    import time

    state = _state(last_run_at=time.time() - 600, throttle_seconds=300)
    saved, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()
    assert saved[0].telegram.chats["123"].live_location.last_run_at == pytest.approx(
        time.time(), abs=5
    )


@pytest.mark.asyncio
async def test_custom_throttle_is_respected(telegram_interface):
    import time

    state = _state(last_run_at=time.time() - 90, throttle_seconds=60)
    _, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_active_run_bypasses_throttle_without_stamping_last_run_at(telegram_interface):
    """An active run takes the update as a cheap injection, so the throttle does not apply."""
    import time

    last_run = time.time()
    state = _state(last_run_at=last_run, throttle_seconds=300)
    run_registry.set("tg-123-0", AgentRun())

    saved, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()  # run() routes it to inject_message
    assert saved[0].telegram.chats["123"].live_location.last_run_at == last_run


@pytest.mark.asyncio
async def test_finished_run_is_not_treated_as_active(telegram_interface):
    import time

    run = AgentRun()
    run.active = False
    run_registry.set("tg-123-0", run)
    state = _state(last_run_at=time.time(), throttle_seconds=300)

    _, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_not_awaited()  # throttled, since there is no live run to inject into


@pytest.mark.asyncio
async def test_stopping_run_is_not_treated_as_active(telegram_interface):
    import time

    run = AgentRun()
    run.request_stop()
    run_registry.set("tg-123-0", run)
    state = _state(last_run_at=time.time(), throttle_seconds=300)

    _, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_disallowed_user_is_ignored(telegram_interface):
    state = _state()
    saved: list = []
    with (
        patch("aug.api.interfaces.telegram.interface.is_allowed", return_value=False),
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
        patch.object(telegram_interface, "run", new=AsyncMock()) as run_mock,
    ):
        await telegram_interface._handle_live_location(_edited_location(), MagicMock())

    assert not saved
    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_edited_message_without_location_is_ignored(telegram_interface):
    update = Update(update_id=1, edited_message=_message(text="hello"))
    saved: list = []
    with (
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
        patch.object(telegram_interface, "run", new=AsyncMock()) as run_mock,
    ):
        await telegram_interface._handle_live_location(update, MagicMock())

    assert not saved
    run_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# receive_message accepts edited location messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_message_reads_edited_location(telegram_interface):
    from aug.api.interfaces.base import LocationContent

    update = _edited_location(lat=10.0, lon=20.0)
    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=_state()):
        incoming = await telegram_interface.receive_message(update)

    assert incoming is not None
    assert incoming.parts == [LocationContent(latitude=10.0, longitude=20.0)]
    assert incoming.thread_id == "tg-123-0"


@pytest.mark.asyncio
async def test_receive_message_returns_none_without_any_message(telegram_interface):
    assert await telegram_interface.receive_message(Update(update_id=1)) is None


# ---------------------------------------------------------------------------
# /throttle command
# ---------------------------------------------------------------------------


def _command_update() -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = CHAT_ID
    update.effective_user.id = USER.id
    update.effective_message.reply_text = AsyncMock()
    return update


async def _throttle(interface, args, state):
    update = _command_update()
    ctx = MagicMock()
    ctx.args = args
    saved: list[AppState] = []
    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
    ):
        await interface._handle_throttle(update, ctx)
    return update.effective_message.reply_text.call_args[0][0], saved


@pytest.mark.asyncio
async def test_throttle_without_args_shows_current_value(telegram_interface):
    reply, saved = await _throttle(telegram_interface, [], _state(throttle_seconds=600))
    assert "600s" in reply
    assert not saved


@pytest.mark.asyncio
async def test_throttle_shows_default_for_unknown_chat(telegram_interface):
    reply, _ = await _throttle(telegram_interface, [], AppState())
    assert "300s" in reply


@pytest.mark.asyncio
async def test_throttle_sets_value(telegram_interface):
    state = _state()
    reply, saved = await _throttle(telegram_interface, ["900"], state)

    assert "900s" in reply
    assert saved[0].telegram.chats["123"].live_location.throttle_seconds == 900


@pytest.mark.asyncio
async def test_throttle_rejects_values_below_minimum(telegram_interface):
    reply, saved = await _throttle(telegram_interface, ["30"], _state())
    assert "Minimum" in reply
    assert not saved


@pytest.mark.asyncio
async def test_throttle_rejects_non_numeric(telegram_interface):
    reply, saved = await _throttle(telegram_interface, ["soon"], _state())
    assert "Usage" in reply
    assert not saved


@pytest.mark.asyncio
async def test_clear_preserves_live_location(telegram_interface):
    """/clear bumps the session counter — it must not wipe a live location in progress."""
    update = _command_update()
    update.message.message_thread_id = None
    update.message.reply_text = AsyncMock()
    state = _state(latitude=51.5, throttle_seconds=600)
    saved: list[AppState] = []

    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
    ):
        await telegram_interface._handle_clear(update, MagicMock())

    live = saved[0].telegram.chats["123"].live_location
    assert saved[0].telegram.chats["123"].session == 1
    assert (live.latitude, live.throttle_seconds) == (51.5, 600)


# ---------------------------------------------------------------------------
# End-to-end through the real BaseInterface.run() routing
# ---------------------------------------------------------------------------


_GEOCODED = "User's current location:\nAddress: Somewhere\nCoordinates: 51.5, -0.12"


@pytest.mark.asyncio
async def test_active_run_receives_the_location_as_an_injection(telegram_interface):
    """With a run in flight, the update lands in its injection queue — no second run."""
    import time

    run = AgentRun()
    run_registry.set("tg-123-0", run)
    state = _state(last_run_at=time.time(), throttle_seconds=300)  # would be throttled

    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state"),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch("aug.api.interfaces.base._geocode", new=AsyncMock(return_value=_GEOCODED)),
        patch.object(telegram_interface, "_execute_run", new=AsyncMock()) as execute,
    ):
        await telegram_interface._handle_live_location(
            _edited_location(lat=51.5, lon=-0.12), MagicMock()
        )

    assert run.pending_agent_injection.get_nowait() == _GEOCODED
    execute.assert_not_awaited()
    assert run_registry.get("tg-123-0") is run


@pytest.mark.asyncio
async def test_unthrottled_update_starts_a_real_run_with_the_location(telegram_interface):
    import asyncio
    import time

    state = _state(last_run_at=time.time() - 600, throttle_seconds=300)

    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state"),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch("aug.api.interfaces.base._geocode", new=AsyncMock(return_value=_GEOCODED)),
        patch.object(telegram_interface, "_execute_run", new=AsyncMock()) as execute,
    ):
        await telegram_interface._handle_live_location(
            _edited_location(lat=51.5, lon=-0.12), MagicMock()
        )
        await asyncio.sleep(0.3)  # let the debounce window elapse

    execute.assert_awaited_once()
    _run, incoming, content, _ctx = execute.await_args[0]
    assert content == _GEOCODED
    assert incoming.thread_id == "tg-123-0"


@pytest.mark.asyncio
async def test_timedelta_live_period_is_converted_to_seconds(telegram_interface):
    """PTB switches live_period to timedelta under PTB_TIMEDELTA=true."""
    from datetime import timedelta

    update = _edited_location()
    object.__setattr__(update.edited_message.location, "_live_period", timedelta(hours=2))
    saved, _ = await _handle(telegram_interface, update, _state())

    live = saved[0].telegram.chats["123"].live_location
    assert live.live_until == pytest.approx(live.updated_at + 7200)
