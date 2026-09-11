"""Typed runtime state backed by data/state.json.

Stores values written by the app itself (counters, scheduler timestamps).
Not for user-facing configuration — use aug/utils/file_settings.py for that.

    from aug.utils.state import load_state, save_state

    st = load_state()
    session = st.telegram.chats.get(chat_id, TelegramChatState()).session

    st = load_state()
    st.telegram.chats[chat_id] = TelegramChatState(session=n + 1)
    save_state(st)
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from aug.utils.data import read_data_file, write_data_file

_STATE_FILE = "state.json"


class LiveLocationState(BaseModel):
    """Latest shared live location for a chat, plus its agent-run throttle."""

    model_config = ConfigDict(extra="ignore")

    latitude: float = 0.0
    longitude: float = 0.0
    updated_at: float = 0.0  # unix timestamp of the last coordinate update
    live_until: float = 0.0  # unix timestamp when the Telegram live_period expires
    last_run_at: float = 0.0  # unix timestamp of the last agent run triggered by a location
    throttle_seconds: int = 300  # minimum seconds between location-triggered agent runs


class TelegramChatState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session: int = 0
    live_location: LiveLocationState = LiveLocationState()


class TelegramState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chats: dict[str, TelegramChatState] = {}


class ConsolidationState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    last_light_run: str | None = None
    last_deep_run: str | None = None


class AppState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    telegram: TelegramState = TelegramState()
    consolidation: ConsolidationState = ConsolidationState()


def load_state() -> AppState:
    """Load runtime state from data/state.json, filling in defaults for any missing fields."""
    raw = read_data_file(_STATE_FILE)
    if not raw:
        return AppState()
    return AppState.model_validate_json(raw)


def save_state(state: AppState) -> None:
    """Persist runtime state to data/state.json."""
    write_data_file(_STATE_FILE, json.dumps(state.model_dump(), indent=2))
