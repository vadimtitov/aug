"""Tests for live location tracking — the BaseInterface store and the Telegram handlers."""

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Location, Message, Update, User
from telegram.ext import filters

from aug.api.interfaces.base import LocationContent
from aug.core.run import AgentRun, run_registry
from aug.utils.state import AppState, ConversationLocationState, LiveLocationState

CHAT_ID = 123
CONVERSATION = "tg-123"  # what get_conversation_id("tg-123-0") resolves to
USER = User(id=7, first_name="V", is_bot=False)
OTHER_USER = User(id=8, first_name="W", is_bot=False)
CHAT = Chat(id=CHAT_ID, type=Chat.PRIVATE)
SENT = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)  # when a share's original message went out
SENT_TS = SENT.timestamp()


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


def _message(user=USER, date=None, **kwargs) -> Message:
    msg = Message(message_id=1, date=date, chat=CHAT, from_user=user, **kwargs)
    msg.set_bot(MagicMock())
    return msg


def _edited_location(
    lat=1.5, lon=2.5, live_period=None, topic_id=None, user=USER, date=None, edit_date=None
) -> Update:
    return Update(
        update_id=1,
        edited_message=_message(
            user=user,
            date=date,
            edit_date=edit_date,
            location=Location(latitude=lat, longitude=lon, live_period=live_period),
            message_thread_id=topic_id,
        ),
    )


def _new_location(lat=1.5, lon=2.5, live_period=None, user=USER, date=None) -> Update:
    return Update(
        update_id=1,
        message=_message(
            user=user,
            date=date,
            location=Location(latitude=lat, longitude=lon, live_period=live_period),
        ),
    )


def _state(users: dict | None = None, **conversation_kwargs) -> AppState:
    """An AppState with a location record for CONVERSATION."""
    state = AppState()
    state.locations[CONVERSATION] = ConversationLocationState(
        users={
            uid: LiveLocationState(user_id=uid, **fields) for uid, fields in (users or {}).items()
        },
        **conversation_kwargs,
    )
    return state


def _stored(state: AppState, user_id: str = "7") -> LiveLocationState:
    return state.locations[CONVERSATION].users[user_id]


async def _handle(interface, update, state):
    """Run _handle_live_location against *state*, returning (saved_states, run_mock)."""
    saved: list[AppState] = []
    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch.object(interface, "run", new=AsyncMock()) as run_mock,
    ):
        await interface._handle_live_location(update, MagicMock())
    return saved, run_mock


# ---------------------------------------------------------------------------
# State: backward compatibility and round-trip
# ---------------------------------------------------------------------------


def test_old_state_file_without_locations_still_loads():
    from aug.utils.state import load_state

    raw = json.dumps({"telegram": {"chats": {"123": {"session": 42}}}})
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()

    assert s.telegram.chats["123"].session == 42
    assert s.locations == {}


def test_state_file_from_the_single_location_model_is_dropped_not_fatal():
    """The old per-chat live_location field no longer exists — extra="ignore" drops it."""
    from aug.utils.state import load_state

    raw = json.dumps(
        {"telegram": {"chats": {"123": {"session": 1, "live_location": {"latitude": 51.5}}}}}
    )
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()

    assert s.telegram.chats["123"].session == 1
    assert s.locations == {}


def test_location_defaults_are_not_shared_between_conversations():
    state = AppState()
    state.locations["a"] = ConversationLocationState()
    state.locations["b"] = ConversationLocationState()
    state.locations["a"].users["7"] = LiveLocationState(user_id="7", latitude=10.0)

    assert state.locations["b"].users == {}


def test_save_round_trips_locations():
    from aug.utils.state import save_state

    written: list[str] = []
    s = _state(
        users={"7": {"latitude": 1.5, "longitude": 2.5, "updated_at": 99.0}},
        throttle_seconds=600,
    )

    with patch("aug.utils.state.write_data_file", side_effect=lambda _f, d: written.append(d)):
        save_state(s)

    loaded = AppState.model_validate_json(written[0])
    assert _stored(loaded).latitude == 1.5
    assert _stored(loaded).user_id == "7"
    assert loaded.locations[CONVERSATION].throttle_seconds == 600


def test_is_live_and_age_seconds():
    now = 1000.0
    live = LiveLocationState(user_id="7", updated_at=now - 30, live_until=now + 60)

    assert live.is_live(now)
    assert not live.is_live(now + 120)
    assert live.age_seconds(now) == 30


def test_a_location_with_no_live_period_is_never_live():
    assert not LiveLocationState(user_id="7", updated_at=1000.0).is_live(1000.0)


# ---------------------------------------------------------------------------
# BaseInterface location store
# ---------------------------------------------------------------------------


async def _record(interface, state, user_id, location, thread_id="tg-123-0"):
    saved: list[AppState] = []
    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
    ):
        interface.record_location(thread_id, user_id, location)
    return saved


@pytest.mark.asyncio
async def test_record_location_creates_the_conversation_entry(telegram_interface):
    state = AppState()
    saved = await _record(
        telegram_interface, state, "7", LocationContent(latitude=51.5, longitude=-0.12)
    )

    live = _stored(saved[0])
    assert (live.latitude, live.longitude, live.user_id) == (51.5, -0.12, "7")
    assert live.updated_at > 0


@pytest.mark.asyncio
async def test_two_users_are_tracked_independently(telegram_interface):
    state = AppState()
    await _record(telegram_interface, state, "7", LocationContent(latitude=1.0, longitude=1.0))
    saved = await _record(
        telegram_interface, state, "8", LocationContent(latitude=2.0, longitude=2.0)
    )

    users = saved[0].locations[CONVERSATION].users
    assert {uid: (loc.latitude, loc.longitude) for uid, loc in users.items()} == {
        "7": (1.0, 1.0),
        "8": (2.0, 2.0),
    }


@pytest.mark.asyncio
async def test_record_location_overwrites_the_same_user(telegram_interface):
    state = _state(users={"7": {"latitude": 1.0, "longitude": 1.0}})
    saved = await _record(
        telegram_interface, state, "7", LocationContent(latitude=9.0, longitude=9.0)
    )

    assert len(saved[0].locations[CONVERSATION].users) == 1
    assert (_stored(saved[0]).latitude, _stored(saved[0]).longitude) == (9.0, 9.0)


@pytest.mark.asyncio
async def test_live_period_sets_expiry(telegram_interface):
    state = AppState()
    saved = await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=1.0, longitude=1.0, live_period=3600),
    )

    live = _stored(saved[0])
    assert live.live_until == pytest.approx(live.updated_at + 3600)
    assert live.is_live()


@pytest.mark.asyncio
async def test_expiry_is_anchored_to_the_send_date_not_to_receipt(telegram_interface):
    """live_period counts from when the message was sent, however late it reaches us."""
    state = AppState()
    saved = await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=1.0, longitude=1.0, live_period=3600, sent_at=SENT_TS),
    )

    assert _stored(saved[0]).live_until == SENT_TS + 3600


@pytest.mark.asyncio
async def test_repeated_updates_do_not_extend_the_share(telegram_interface):
    """Every edit restates the same period against the same send date — no creep."""
    state = AppState()
    sent = time.time() - 1800  # half an hour into a one-hour share
    for offset in (0, 60, 120):
        saved = await _record(
            telegram_interface,
            state,
            "7",
            LocationContent(
                latitude=1.0,
                longitude=1.0,
                live_period=3600,
                sent_at=sent,
                reported_at=sent + 1800 + offset,
            ),
        )

    assert _stored(saved[0]).live_until == sent + 3600


@pytest.mark.asyncio
async def test_a_share_delivered_after_it_expired_is_not_revived(telegram_interface):
    state = AppState()
    saved = await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=1.0, longitude=1.0, live_period=600, sent_at=time.time() - 3600),
    )

    assert not _stored(saved[0]).is_live()


@pytest.mark.asyncio
async def test_missing_live_period_keeps_a_running_expiry(telegram_interface):
    """A live edit that does not restate the period must not cut the share short."""
    live_until = time.time() + 3600
    state = _state(users={"7": {"live_until": live_until}})
    saved = await _record(
        telegram_interface, state, "7", LocationContent(latitude=1.0, longitude=1.0)
    )

    assert _stored(saved[0]).live_until == live_until


@pytest.mark.asyncio
async def test_a_static_pin_clears_a_finished_share(telegram_interface):
    """An expired live_until must not linger and make a one-off pin look live."""
    state = _state(users={"7": {"live_until": time.time() - 60}})
    saved = await _record(
        telegram_interface, state, "7", LocationContent(latitude=1.0, longitude=1.0)
    )

    assert _stored(saved[0]).live_until == 0.0


# ---------------------------------------------------------------------------
# Out-of-order delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_late_update_does_not_replace_a_newer_position(telegram_interface):
    now = time.time()
    state = AppState()
    await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=2.0, longitude=2.0, reported_at=now),
    )
    saved = await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=1.0, longitude=1.0, reported_at=now - 60),
    )

    assert not saved  # nothing written — the stored position is the newer one
    assert state.locations[CONVERSATION].users["7"].latitude == 2.0


@pytest.mark.asyncio
async def test_a_newer_update_replaces_the_stored_position(telegram_interface):
    now = time.time()
    state = _state(users={"7": {"latitude": 1.0, "reported_at": now - 60}})
    saved = await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=2.0, longitude=2.0, reported_at=now),
    )

    assert (_stored(saved[0]).latitude, _stored(saved[0]).reported_at) == (2.0, now)


@pytest.mark.asyncio
async def test_a_late_update_for_one_user_leaves_the_others_alone(telegram_interface):
    now = time.time()
    state = _state(users={"7": {"latitude": 2.0, "reported_at": now}})
    await _record(
        telegram_interface,
        state,
        "8",
        LocationContent(latitude=5.0, longitude=5.0, reported_at=now - 60),
    )

    users = state.locations[CONVERSATION].users
    assert users["7"].latitude == 2.0
    assert users["8"].latitude == 5.0  # a different user has no stored order to violate


@pytest.mark.asyncio
async def test_updates_without_timestamps_are_always_accepted(telegram_interface):
    """An interface that supplies no platform time keeps the old last-write-wins rule."""
    state = _state(users={"7": {"latitude": 1.0, "reported_at": time.time()}})
    saved = await _record(
        telegram_interface, state, "7", LocationContent(latitude=9.0, longitude=9.0)
    )

    assert _stored(saved[0]).latitude == 9.0


def test_age_seconds_measures_from_the_reported_time(telegram_interface):
    """The position's own timestamp is what makes it stale, not when we wrote it down."""
    now = 1000.0
    live = LiveLocationState(user_id="7", updated_at=now, reported_at=now - 45)

    assert live.age_seconds(now) == 45


@pytest.mark.asyncio
async def test_live_locations_reads_back_what_was_recorded(telegram_interface):
    state = AppState()
    await _record(telegram_interface, state, "7", LocationContent(latitude=51.5, longitude=-0.12))

    with patch("aug.api.interfaces.base.load_state", return_value=state):
        locations = telegram_interface.live_locations("tg-123-0")

    assert list(locations) == ["7"]
    assert (locations["7"].latitude, locations["7"].longitude) == (51.5, -0.12)


def test_live_locations_is_empty_for_an_unknown_conversation(telegram_interface):
    with patch("aug.api.interfaces.base.load_state", return_value=AppState()):
        assert telegram_interface.live_locations("tg-999-0") == {}


@pytest.mark.asyncio
async def test_location_state_survives_a_new_session(telegram_interface):
    """Keying by conversation, not thread, means /clear cannot lose a live location."""
    state = AppState()
    await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=51.5, longitude=-0.12),
        thread_id="tg-123-0",
    )

    with patch("aug.api.interfaces.base.load_state", return_value=state):
        assert "7" in telegram_interface.live_locations("tg-123-1")


@pytest.mark.asyncio
async def test_topics_do_not_share_location_state(telegram_interface):
    state = AppState()
    await _record(
        telegram_interface,
        state,
        "7",
        LocationContent(latitude=1.0, longitude=1.0),
        thread_id="tg-123-topic-5",
    )

    with patch("aug.api.interfaces.base.load_state", return_value=state):
        assert telegram_interface.live_locations("tg-123-topic-6") == {}


def test_claim_location_run_stamps_and_then_throttles(telegram_interface):
    state = _state(throttle_seconds=300)
    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state"),
    ):
        assert telegram_interface.claim_location_run("tg-123-0") is True
        assert telegram_interface.claim_location_run("tg-123-0") is False

    assert state.locations[CONVERSATION].last_run_at == pytest.approx(time.time(), abs=5)


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
    assert _matching_handlers(bot_app, _new_location()) == ["_handle_input"]


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
    new = _new_location()

    assert edited_only.check_update(edited) and not new_only.check_update(edited)
    assert new_only.check_update(new) and not edited_only.check_update(new)


# ---------------------------------------------------------------------------
# _handle_live_location behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinates_are_persisted(telegram_interface):
    state = _state(last_run_at=0.0)
    saved, _ = await _handle(telegram_interface, _edited_location(lat=51.5, lon=-0.12), state)

    live = _stored(saved[0])
    assert (live.latitude, live.longitude) == (51.5, -0.12)
    assert live.updated_at > 0


@pytest.mark.asyncio
async def test_update_is_keyed_by_sender_not_chat(telegram_interface):
    """Two people sharing in one chat must not overwrite each other."""
    state = _state(last_run_at=0.0)
    await _handle(telegram_interface, _edited_location(lat=1.0, user=USER), state)
    saved, _ = await _handle(telegram_interface, _edited_location(lat=2.0, user=OTHER_USER), state)

    users = saved[0].locations[CONVERSATION].users
    assert (users["7"].latitude, users["8"].latitude) == (1.0, 2.0)


@pytest.mark.asyncio
async def test_live_period_from_an_edit_sets_expiry(telegram_interface):
    saved, _ = await _handle(telegram_interface, _edited_location(live_period=3600), _state())

    live = _stored(saved[0])
    assert live.live_until == pytest.approx(live.updated_at + 3600)


@pytest.mark.asyncio
async def test_throttled_update_saves_but_does_not_run(telegram_interface):
    state = _state(last_run_at=time.time(), throttle_seconds=300)
    saved, run_mock = await _handle(telegram_interface, _edited_location(lat=9.0), state)

    assert _stored(saved[0]).latitude == 9.0
    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_elapsed_throttle_starts_a_run_and_stamps_last_run_at(telegram_interface):
    state = _state(last_run_at=time.time() - 600, throttle_seconds=300)
    saved, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()
    assert saved[-1].locations[CONVERSATION].last_run_at == pytest.approx(time.time(), abs=5)


@pytest.mark.asyncio
async def test_custom_throttle_is_respected(telegram_interface):
    state = _state(last_run_at=time.time() - 90, throttle_seconds=60)
    _, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_one_users_run_throttles_the_whole_conversation(telegram_interface):
    """The throttle limits how often the agent is woken for a thread, whoever shares."""
    state = _state(last_run_at=0.0, throttle_seconds=300)
    _, first = await _handle(telegram_interface, _edited_location(user=USER), state)
    saved, second = await _handle(telegram_interface, _edited_location(user=OTHER_USER), state)

    first.assert_awaited_once()
    second.assert_not_awaited()
    assert set(saved[-1].locations[CONVERSATION].users) == {"7", "8"}  # still recorded


@pytest.mark.asyncio
async def test_active_run_bypasses_throttle_without_stamping_last_run_at(telegram_interface):
    """An active run takes the update as a cheap injection, so the throttle does not apply."""
    last_run = time.time()
    state = _state(last_run_at=last_run, throttle_seconds=300)
    run_registry.set("tg-123-0", AgentRun())

    saved, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_awaited_once()  # run() routes it to inject_message
    assert saved[-1].locations[CONVERSATION].last_run_at == last_run


@pytest.mark.asyncio
async def test_finished_run_is_not_treated_as_active(telegram_interface):
    run = AgentRun()
    run.active = False
    run_registry.set("tg-123-0", run)
    state = _state(last_run_at=time.time(), throttle_seconds=300)

    _, run_mock = await _handle(telegram_interface, _edited_location(), state)

    run_mock.assert_not_awaited()  # throttled, since there is no live run to inject into


@pytest.mark.asyncio
async def test_stopping_run_is_not_treated_as_active(telegram_interface):
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
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
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
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
        patch.object(telegram_interface, "run", new=AsyncMock()) as run_mock,
    ):
        await telegram_interface._handle_live_location(update, MagicMock())

    assert not saved
    run_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# receive_message translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_message_reads_edited_location(telegram_interface):
    update = _edited_location(lat=10.0, lon=20.0, live_period=600)
    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=_state()):
        incoming = await telegram_interface.receive_message(update)

    assert incoming is not None
    assert incoming.parts == [
        LocationContent(latitude=10.0, longitude=20.0, live_period=600, sender_name="V")
    ]
    assert incoming.thread_id == "tg-123-0"
    assert incoming.user_id == "7"
    assert incoming.sender_id == "123"  # the chat, which is where a reply goes


@pytest.mark.asyncio
async def test_receive_message_returns_none_without_any_message(telegram_interface):
    assert await telegram_interface.receive_message(Update(update_id=1)) is None


@pytest.mark.asyncio
async def test_timedelta_live_period_is_converted_to_seconds(telegram_interface):
    """PTB switches live_period to timedelta under PTB_TIMEDELTA=true."""
    from datetime import timedelta

    update = _edited_location()
    object.__setattr__(update.edited_message.location, "_live_period", timedelta(hours=2))
    saved, _ = await _handle(telegram_interface, update, _state())

    live = _stored(saved[0])
    assert live.live_until == pytest.approx(live.updated_at + 7200)


# ---------------------------------------------------------------------------
# /throttle command
# ---------------------------------------------------------------------------


def _command_update() -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = CHAT_ID
    update.effective_user.id = USER.id
    update.effective_message.message_thread_id = None
    update.effective_message.reply_text = AsyncMock()
    return update


async def _throttle(interface, args, state):
    update = _command_update()
    ctx = MagicMock()
    ctx.args = args
    saved: list[AppState] = []
    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
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
    reply, saved = await _throttle(telegram_interface, ["900"], _state())

    assert "900s" in reply
    assert saved[0].locations[CONVERSATION].throttle_seconds == 900


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
    state = _state(users={"7": {"latitude": 51.5}}, throttle_seconds=600)
    saved: list[AppState] = []

    with (
        patch("aug.api.interfaces.telegram.interface.load_state", return_value=state),
        patch("aug.api.interfaces.telegram.interface.save_state", side_effect=saved.append),
    ):
        await telegram_interface._handle_clear(update, MagicMock())

    assert saved[0].telegram.chats["123"].session == 1
    assert _stored(saved[0]).latitude == 51.5
    assert saved[0].locations[CONVERSATION].throttle_seconds == 600


# ---------------------------------------------------------------------------
# End-to-end through the real BaseInterface.run() routing
# ---------------------------------------------------------------------------


_GEOCODED = "User's current location:\nAddress: Somewhere\nCoordinates: 51.5, -0.12"


@pytest.mark.asyncio
async def test_initial_share_is_saved_on_the_way_through_run(telegram_interface):
    """The first share arrives as a normal message — it must land in the store too."""
    state = AppState()
    saved: list[AppState] = []

    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch("aug.api.interfaces.base._geocode", new=AsyncMock(return_value=_GEOCODED)),
        patch.object(telegram_interface, "_execute_run", new=AsyncMock()),
    ):
        await telegram_interface.run(_new_location(lat=51.5, lon=-0.12, live_period=900))

    live = _stored(saved[0])
    assert (live.latitude, live.longitude) == (51.5, -0.12)
    assert live.updated_at > 0
    assert live.live_until == pytest.approx(live.updated_at + 900)
    assert live.is_live()


@pytest.mark.asyncio
async def test_active_run_receives_the_location_as_an_injection(telegram_interface):
    """With a run in flight, the update lands in its injection queue — no second run."""
    run = AgentRun()
    run_registry.set("tg-123-0", run)
    state = _state(last_run_at=time.time(), throttle_seconds=300)  # would be throttled

    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state"),
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

    state = _state(last_run_at=time.time() - 600, throttle_seconds=300)

    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state"),
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
    assert incoming.user_id == "7"


# ---------------------------------------------------------------------------
# The throttle is stamped where the run actually starts
# ---------------------------------------------------------------------------


async def _run_update(interface, update, state, incoming=None):
    """Drive the real BaseInterface.run() for *update*, returning the saved states."""
    import asyncio

    saved: list[AppState] = []
    patches = [
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state", side_effect=saved.append),
        patch("aug.api.interfaces.telegram.utils.load_state", return_value=state),
        patch("aug.api.interfaces.base._geocode", new=AsyncMock(return_value=_GEOCODED)),
        patch.object(interface, "_execute_run", new=AsyncMock()),
    ]
    if incoming is not None:
        patches.append(patch.object(interface, "receive_message", AsyncMock(return_value=incoming)))
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        if incoming is not None:
            with patches[5]:
                await interface.run(update)
        else:
            await interface.run(update)
        await asyncio.sleep(0.3)  # let the debounce window elapse
    return saved


@pytest.mark.asyncio
async def test_initial_share_consumes_the_throttle(telegram_interface):
    """A first share wakes the agent, so the update a minute later must not."""
    state = _state(last_run_at=0.0, throttle_seconds=300)

    await _run_update(telegram_interface, _new_location(lat=51.5, lon=-0.12), state)

    assert state.locations[CONVERSATION].last_run_at == pytest.approx(time.time(), abs=5)
    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state"),
    ):
        assert telegram_interface.claim_location_run("tg-123-0") is False


@pytest.mark.asyncio
async def test_a_run_started_after_the_gate_still_stamps_the_throttle(telegram_interface):
    """The gate saw an active run; it finished during preprocessing and run() started a
    fresh one. The throttle has to be consumed there, not left untouched."""
    state = _state(last_run_at=0.0, throttle_seconds=300)
    stale = AgentRun()
    stale.active = False
    run_registry.set("tg-123-0", stale)

    await _run_update(telegram_interface, _edited_location(lat=51.5, lon=-0.12), state)

    assert state.locations[CONVERSATION].last_run_at == pytest.approx(time.time(), abs=5)


@pytest.mark.asyncio
async def test_an_injected_location_does_not_consume_the_throttle(telegram_interface):
    """Injecting into a running agent is free, so it must not push the next wake out."""
    state = _state(last_run_at=0.0, throttle_seconds=300)
    run_registry.set("tg-123-0", AgentRun())

    await _run_update(telegram_interface, _edited_location(lat=51.5, lon=-0.12), state)

    assert state.locations[CONVERSATION].last_run_at == 0.0


@pytest.mark.asyncio
async def test_a_message_without_a_location_never_stamps_the_throttle(telegram_interface):
    state = _state(last_run_at=0.0, throttle_seconds=300)
    update = Update(update_id=1, message=_message(text="hello"))

    await _run_update(telegram_interface, update, state)

    assert state.locations[CONVERSATION].last_run_at == 0.0


# ---------------------------------------------------------------------------
# Sender fallback when an interface has no per-user id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_location_without_a_user_id_is_keyed_by_sender(telegram_interface):
    """An interface with no per-user identity must not pile everyone under one key."""
    from aug.api.interfaces.base import IncomingMessage

    state = _state()
    incoming = IncomingMessage(
        parts=[LocationContent(latitude=1.0, longitude=2.0)],
        interface="rest_api",
        sender_id="rest-caller",
        thread_id="tg-123-0",
        agent_version="fake",
    )

    await _run_update(telegram_interface, MagicMock(), state, incoming=incoming)

    assert list(state.locations[CONVERSATION].users) == ["rest-caller"]


# ---------------------------------------------------------------------------
# Telegram timestamp extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_message_carries_the_send_and_edit_dates(telegram_interface):
    edited = SENT + timedelta(minutes=5)
    update = _edited_location(live_period=3600, date=SENT, edit_date=edited)

    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=_state()):
        incoming = await telegram_interface.receive_message(update)

    part = incoming.parts[0]
    assert part.sent_at == SENT_TS  # anchors the expiry
    assert part.reported_at == edited.timestamp()  # orders the updates


@pytest.mark.asyncio
async def test_an_unedited_location_reports_at_its_send_date(telegram_interface):
    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=_state()):
        incoming = await telegram_interface.receive_message(_new_location(date=SENT))

    assert incoming.parts[0].reported_at == SENT_TS


@pytest.mark.asyncio
async def test_successive_edits_of_one_share_expire_at_the_same_moment(telegram_interface):
    """End to end: two edits an hour into a 3h share land on one expiry, not two."""
    state = _state()
    for minutes in (60, 120):
        await _handle(
            telegram_interface,
            _edited_location(
                live_period=10800, date=SENT, edit_date=SENT + timedelta(minutes=minutes)
            ),
            state,
        )

    assert _stored(state).live_until == SENT_TS + 10800


# ---------------------------------------------------------------------------
# A stale position is kept away from the agent, not just out of storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_location_reports_whether_it_accepted_the_position(telegram_interface):
    now = time.time()
    state = AppState()

    with (
        patch("aug.api.interfaces.base.load_state", return_value=state),
        patch("aug.api.interfaces.base.save_state"),
    ):
        fresh = telegram_interface.record_location(
            "tg-123-0", "7", LocationContent(latitude=1.0, longitude=1.0, reported_at=now)
        )
        stale = telegram_interface.record_location(
            "tg-123-0", "7", LocationContent(latitude=2.0, longitude=2.0, reported_at=now - 60)
        )

    assert (fresh, stale) == (True, False)


@pytest.mark.asyncio
async def test_a_stale_update_never_reaches_the_agent(telegram_interface):
    """Dropping it from storage is not enough — the agent must not be told either."""
    now = time.time()
    state = _state(users={"7": {"latitude": 2.0, "reported_at": now}}, last_run_at=0.0)

    saved, run_mock = await _handle(
        telegram_interface,
        _edited_location(lat=1.0, date=SENT, edit_date=datetime.fromtimestamp(now - 60, tz=UTC)),
        state,
    )

    run_mock.assert_not_awaited()
    assert not saved
    assert state.locations[CONVERSATION].last_run_at == 0.0  # throttle untouched
    assert _stored(state).latitude == 2.0


@pytest.mark.asyncio
async def test_a_fresh_update_still_reaches_the_agent(telegram_interface):
    now = time.time()
    state = _state(users={"7": {"latitude": 2.0, "reported_at": now - 60}}, last_run_at=0.0)

    _, run_mock = await _handle(
        telegram_interface,
        _edited_location(lat=1.0, date=SENT, edit_date=datetime.fromtimestamp(now, tz=UTC)),
        state,
    )

    run_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stale_location_does_not_consume_the_throttle_in_run(telegram_interface):
    """run() is reached by other paths too — a dropped position must not stamp there."""
    now = time.time()
    state = _state(users={"7": {"latitude": 2.0, "reported_at": now}}, last_run_at=0.0)

    await _run_update(
        telegram_interface,
        _new_location(lat=1.0, date=datetime.fromtimestamp(now - 60, tz=UTC)),
        state,
    )

    assert state.locations[CONVERSATION].last_run_at == 0.0


# ---------------------------------------------------------------------------
# Sender attribution in the text the agent reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_message_names_the_sharer(telegram_interface):
    sharer = User(id=9, first_name="Vadim", last_name="K", is_bot=False)

    with patch("aug.api.interfaces.telegram.utils.load_state", return_value=_state()):
        incoming = await telegram_interface.receive_message(_new_location(user=sharer))

    assert incoming.parts[0].sender_name == "Vadim K"


@pytest.mark.asyncio
async def test_geocoded_text_names_the_sharer():
    from aug.api.interfaces.base import _preprocess

    with patch("aug.api.interfaces.base.httpx.AsyncClient") as client:
        response = MagicMock()
        response.json.return_value = {"display_name": "10 Downing St, London"}
        client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        text = await _preprocess(
            [LocationContent(latitude=51.5, longitude=-0.12, sender_name="Vadim")]
        )

    assert text.startswith("Vadim's current location:")
    assert "10 Downing St, London" in text
    assert "51.5, -0.12" in text


@pytest.mark.asyncio
async def test_geocoded_text_falls_back_to_user_without_a_name():
    """Interfaces that supply no name keep the wording the agent already knows."""
    from aug.api.interfaces.base import _preprocess

    with patch("aug.api.interfaces.base.httpx.AsyncClient") as client:
        response = MagicMock()
        response.json.return_value = {"display_name": "Somewhere"}
        client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        text = await _preprocess([LocationContent(latitude=1.0, longitude=2.0)])

    assert text.startswith("User's current location:")


@pytest.mark.asyncio
async def test_two_sharers_are_distinguishable_in_one_conversation(telegram_interface):
    """The whole point: in a group, the agent can tell whose position is whose."""
    state = _state()
    texts = []
    for user in (
        User(id=7, first_name="V", is_bot=False),
        User(id=8, first_name="W", is_bot=False),
    ):
        with patch("aug.api.interfaces.telegram.utils.load_state", return_value=state):
            incoming = await telegram_interface.receive_message(_new_location(user=user))
        texts.append(incoming.parts[0].sender_name)

    assert texts == ["V", "W"]
