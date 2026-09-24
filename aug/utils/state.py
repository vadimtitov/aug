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

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

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
    updated_at: float = 0.0  # unix timestamp of when we recorded these coordinates
    reported_at: float = 0.0  # unix timestamp the platform put on them (0 = unknown)
    live_until: float = 0.0  # unix timestamp when the live sharing period expires (0 = not live)

    def age_seconds(self, now: float | None = None) -> float:
        """Seconds since this position was reported.

        Measured from the platform's own timestamp where there is one — how stale the
        position is, not how long ago we happened to write it down.
        """
        return (time.time() if now is None else now) - (self.reported_at or self.updated_at)

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


class McpOperation(BaseModel):
    """One install/remove operation, tracked across the restart it triggers.

    A restart that crashes AUG before the next boot can update this record
    leaves it at ``restart_pending`` — that's exactly the state
    ``MCPManager.reconcile_operations`` looks for and resolves on the next
    successful startup, so an operation is never silently lost.

    ``interface``/``thread_id`` name the conversation that requested the
    operation, so its outcome can be delivered back there specifically
    instead of only riding along on the general startup announcement.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    action: Literal["install", "remove"]
    server_name: str
    state: Literal["saved", "restart_pending", "active", "failed"]
    detail: str = ""
    created_at: float = 0.0
    interface: str = ""
    thread_id: str = ""


class McpCredentialBinding(BaseModel):
    """One requested credential input and how the plan proposes to satisfy it."""

    model_config = ConfigDict(extra="ignore")

    # The name the server actually reads at runtime — an env var name for
    # stdio, an HTTP header name for http. Never itself a hushed secret name.
    target_name: str
    # The hushed secret name this will be bound to, e.g. "SENTRY_BEARER_TOKEN".
    secret_name: str
    # Whether `secret_name` already existed in hushed at plan-build time.
    bound: bool


class McpInstallPlan(BaseModel):
    """An immutable, durable snapshot of one install decision.

    Built once, the moment ``install_mcp_server`` first resolves a search
    result to an index, and persisted before the approval interrupt pauses
    the graph — so resuming (even from a different process, after a crash)
    approves and installs exactly what was previewed, never a result a
    later search happened to reorder into that slot.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    thread_id: str
    search_index: int
    server_name: str  # registry name, e.g. "io.github.x/server-postgres"
    slug: str  # config name this will be saved under, e.g. "postgres"
    version: str
    transport: Literal["stdio", "http"]
    command: str = ""
    args: list[str] = []
    url: str = ""
    credentials: list[McpCredentialBinding] = []
    # Literal, non-secret env vars / headers the registry declared with a
    # concrete value or default — see McpRegistryServer.literal_inputs.
    literal_inputs: dict[str, str] = {}
    created_at: float = 0.0


class McpState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operations: list[McpOperation] = []
    # Keyed by thread_id — at most one pending install plan per conversation,
    # since LangGraph fully pauses that thread while it awaits approval.
    install_plans: dict[str, McpInstallPlan] = {}


class AppState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    telegram: TelegramState = TelegramState()
    consolidation: ConsolidationState = ConsolidationState()
    # Keyed by BaseInterface.conversation_id — interface-namespaced, so this is shared
    # by every frontend rather than living under any one of them.
    locations: dict[str, ConversationLocationState] = {}
    mcp: McpState = McpState()


def load_state() -> AppState:
    """Load runtime state from data/state.json, filling in defaults for any missing fields."""
    raw = read_data_file(_STATE_FILE)
    if not raw:
        return AppState()
    return AppState.model_validate_json(raw)


def save_state(state: AppState) -> None:
    """Persist runtime state to data/state.json."""
    write_data_file(_STATE_FILE, json.dumps(state.model_dump(), indent=2))


# Serializes concurrent read-modify-write cycles against state.json — the same
# hazard file_settings.update_settings() guards against, applied here for
# things like McpManager's install-plan bookkeeping, where two conversations
# can otherwise both load a stale snapshot and clobber each other's save.
_state_lock = asyncio.Lock()


@asynccontextmanager
async def update_state() -> AsyncIterator[AppState]:
    """Serialized read-modify-write: reload the current file under a lock, let
    the caller mutate it, then save.

        async with update_state() as st:
            st.mcp.operations.append(...)
    """
    async with _state_lock:
        state = load_state()
        yield state
        save_state(state)
