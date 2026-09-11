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
import time

from pydantic import BaseModel, ConfigDict

from aug.utils.data import read_data_file, write_data_file

_STATE_FILE = "state.json"


class LiveLocationState(BaseModel):
    """The latest location one user shared in one conversation.

    Written by BaseInterface.record_location for every location that arrives —
    the first share as well as each live update — so a reader never sees a gap
    between the two.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str = ""  # platform user id of the sharer, mirrored from the dict key
    latitude: float = 0.0
    longitude: float = 0.0
    updated_at: float = 0.0  # unix timestamp of the last coordinate update
    live_until: float = 0.0  # unix timestamp when the live sharing period expires (0 = not live)

    def age_seconds(self, now: float | None = None) -> float:
        """Seconds since these coordinates were last updated."""
        return (time.time() if now is None else now) - self.updated_at

    def is_live(self, now: float | None = None) -> bool:
        """True while the sharing period the user granted is still running."""
        return self.live_until > (time.time() if now is None else now)


class ConversationLocationState(BaseModel):
    """Every live location shared in one conversation, plus its agent-run throttle.

    Keyed by user id: several people in a group can share at once and are tracked
    independently.  The throttle is per conversation, not per user — it limits how
    often location updates wake the agent for this thread, whoever sent them.
    """

    model_config = ConfigDict(extra="ignore")

    users: dict[str, LiveLocationState] = {}
    last_run_at: float = 0.0  # unix timestamp of the last agent run triggered by a location
    throttle_seconds: int = 300  # minimum seconds between location-triggered agent runs


class TelegramChatState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session: int = 0


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
    # Keyed by BaseInterface.conversation_id — interface-namespaced, so this is shared
    # by every frontend rather than living under any one of them.
    locations: dict[str, ConversationLocationState] = {}


def load_state() -> AppState:
    """Load runtime state from data/state.json, filling in defaults for any missing fields."""
    raw = read_data_file(_STATE_FILE)
    if not raw:
        return AppState()
    return AppState.model_validate_json(raw)


def save_state(state: AppState) -> None:
    """Persist runtime state to data/state.json."""
    write_data_file(_STATE_FILE, json.dumps(state.model_dump(), indent=2))
