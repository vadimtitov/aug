"""Tests for aug/utils/state.py — Pydantic-typed runtime state."""

import json
from unittest.mock import patch

from aug.utils.state import AppState, TelegramChatState, load_state, save_state


def test_load_returns_defaults_when_file_empty():
    with patch("aug.utils.state.read_data_file", return_value=""):
        s = load_state()
    assert s.telegram.chats == {}
    assert s.consolidation.last_light_run is None
    assert s.consolidation.last_deep_run is None


def test_load_session_counter():
    raw = json.dumps({"telegram": {"chats": {"123": {"session": 42}}}})
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()
    assert s.telegram.chats["123"].session == 42


def test_load_missing_chat_returns_default_session():
    with patch("aug.utils.state.read_data_file", return_value=""):
        s = load_state()
    assert s.telegram.chats.get("999", TelegramChatState()).session == 0


def test_load_consolidation_timestamps():
    raw = json.dumps(
        {
            "consolidation": {
                "last_light_run": "2026-04-13T00:24:51+00:00",
                "last_deep_run": "2026-04-13T00:26:53+00:00",
            }
        }
    )
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()
    assert s.consolidation.last_light_run == "2026-04-13T00:24:51+00:00"
    assert s.consolidation.last_deep_run == "2026-04-13T00:26:53+00:00"


def test_load_ignores_unknown_keys():
    raw = json.dumps({"unknown": {"foo": "bar"}})
    with patch("aug.utils.state.read_data_file", return_value=raw):
        s = load_state()
    assert isinstance(s, AppState)


def test_save_round_trips_session():
    written: list[str] = []
    s = AppState()
    s.telegram.chats["99"] = TelegramChatState(session=7)

    with patch("aug.utils.state.write_data_file", side_effect=lambda _f, d: written.append(d)):
        save_state(s)

    loaded = AppState.model_validate_json(written[0])
    assert loaded.telegram.chats["99"].session == 7


def test_save_round_trips_consolidation_timestamps():
    written: list[str] = []
    s = AppState()
    s.consolidation.last_light_run = "2026-04-13T00:24:51+00:00"
    s.consolidation.last_deep_run = "2026-04-13T00:26:53+00:00"

    with patch("aug.utils.state.write_data_file", side_effect=lambda _f, d: written.append(d)):
        save_state(s)

    loaded = AppState.model_validate_json(written[0])
    assert loaded.consolidation.last_light_run == "2026-04-13T00:24:51+00:00"
    assert loaded.consolidation.last_deep_run == "2026-04-13T00:26:53+00:00"


def test_save_writes_to_state_file():
    filename_used: list[str] = []
    s = AppState()

    with patch(
        "aug.utils.state.write_data_file",
        side_effect=lambda f, _d: filename_used.append(f),
    ):
        save_state(s)

    assert filename_used == ["state.json"]
