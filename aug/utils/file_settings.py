"""Typed settings backed by data/settings.json.

All user-facing configuration lives here. Access via:

    from aug.utils.file_settings import load_settings, save_settings

    s = load_settings()
    agent = s.conversations.get(conversation_id, ConversationSettings()).agent

    s = load_settings()
    s.conversations[conversation_id] = ConversationSettings(agent="v2")
    save_settings(s)

Conversation IDs are interface-scoped and stable across context resets — see
``BaseInterface.conversation_id``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from aug.utils.data import read_data_file, write_data_file

_SETTINGS_FILE = "settings.json"

# MCP server config names double as tool-namespace prefixes and directory-safe
# identifiers, so they're restricted the same way env-var-derived slugs are.
_MCP_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
# env/header values are always hushed references, never plaintext — see
# McpServerConfig's docstring.
_HUSHED_REF_RE = re.compile(r"^hushed:[A-Za-z_][A-Za-z0-9_]*$")


class ConversationSettings(BaseModel):
    """Per-conversation settings, keyed by ``BaseInterface.conversation_id``."""

    model_config = ConfigDict(extra="ignore")

    agent: str = "default"


class ConsolidationSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-5.1"


class SshTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    host: str
    port: int = 22
    user: str
    key_path: str
    known_hosts: str = ""
    verify_host: bool = True


class ApprovalRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str = "*"
    target: str = "*"
    pattern: str


class SshToolSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    targets: list[SshTarget] = []
    max_download_bytes: int = 1_073_741_824  # 1 GB


class BashToolSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    blacklist: list[str] = []


class BrowserToolSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-5.1"


class ImageGenToolSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-image-1.5"


class ToolSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ssh: SshToolSettings = SshToolSettings()
    approvals: list[ApprovalRule] = []
    bash: BashToolSettings = BashToolSettings()
    browser: BrowserToolSettings = BrowserToolSettings()
    image_gen: ImageGenToolSettings = ImageGenToolSettings()


class McpServerConfig(BaseModel):
    """One configured MCP server. Loaded by MCPManager at startup.

    ``env`` / ``headers`` values are ``hushed:KEY_NAME`` references, never plaintext
    secrets — see ``aug/core/mcp_manager.py`` for how they're resolved. ``args``
    carries the pinned package version for stdio servers (e.g.
    ``["-y", "@modelcontextprotocol/server-github@1.0.0"]``) so an install never
    silently picks up a newer, unreviewed release on restart.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    transport: Literal["stdio", "http"]
    command: str = ""  # stdio only
    args: list[str] = []  # stdio only
    env: dict[str, str] = {}  # stdio only — hushed:KEY references
    env_static: dict[str, str] = {}  # stdio only — literal, non-secret defaults
    url: str = ""  # http only
    headers: dict[str, str] = {}  # http only — hushed:KEY references
    headers_static: dict[str, str] = {}  # http only — literal, non-secret defaults
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _MCP_NAME_RE.match(v):
            raise ValueError(f"invalid MCP server name {v!r} — must match {_MCP_NAME_RE.pattern}")
        return v

    @field_validator("env", "headers")
    @classmethod
    def _validate_refs(cls, v: dict[str, str]) -> dict[str, str]:
        # Never echo `ref` back in the error — a hand-edited settings.json can
        # easily hold a real plaintext secret here instead of a reference, and
        # that value must not be repeated into a validation error message that
        # ends up in logs. Empty is rejected too: it's neither a valid
        # reference nor a value this model is meant to carry.
        for target_name in v:
            if not _HUSHED_REF_RE.match(v[target_name]):
                raise ValueError(
                    f"invalid credential reference for {target_name!r} — must be 'hushed:NAME'"
                )
        return v

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> McpServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server {self.name!r}: stdio transport requires 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError(f"MCP server {self.name!r}: http transport requires 'url'")
        return self


class HomeAssistantReflexSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_label: str = "aug"


class ReflexSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    homeassistant: HomeAssistantReflexSettings = HomeAssistantReflexSettings()


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversations: dict[str, ConversationSettings] = {}
    consolidation: ConsolidationSettings = ConsolidationSettings()
    tools: ToolSettings = ToolSettings()
    reflexes: ReflexSettings = ReflexSettings()
    mcp_servers: list[McpServerConfig] = []

    @field_validator("mcp_servers")
    @classmethod
    def _unique_mcp_names(cls, v: list[McpServerConfig]) -> list[McpServerConfig]:
        names = [s.name for s in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate MCP server name(s): {', '.join(dupes)}")
        return v


def load_settings() -> AppSettings:
    """Load settings from data/settings.json, filling in defaults for any missing fields."""
    raw = read_data_file(_SETTINGS_FILE)
    if not raw:
        return AppSettings()
    return AppSettings.model_validate(_migrate(json.loads(raw)))


def save_settings(settings: AppSettings) -> None:
    """Persist settings to data/settings.json."""
    write_data_file(_SETTINGS_FILE, json.dumps(settings.model_dump(), indent=2))


# A single process-wide lock serializes every settings read-modify-write so two
# concurrent mutations (e.g. two installs) can never both load a stale snapshot
# and clobber each other's save — see update_settings().
_settings_lock = asyncio.Lock()


@asynccontextmanager
async def update_settings() -> AsyncIterator[AppSettings]:
    """Serialized read-modify-write: reload the current file under a lock, let
    the caller mutate it, then save — so the check that decides whether to
    write (e.g. "is this name already configured?") is never based on a
    snapshot another concurrent writer has since made stale.

        async with update_settings() as s:
            if not any(x.name == name for x in s.mcp_servers):
                s.mcp_servers.append(cfg)
    """
    async with _settings_lock:
        settings = load_settings()
        yield settings
        save_settings(settings)


def _migrate(data: dict) -> dict:
    """Carry the legacy per-chat agent setting into ``conversations``.

    Agent versions used to live under ``telegram.chats.<chat_id>.agent``, which
    made every forum topic in a group share one version.  They now live under
    ``conversations.<conversation_id>``, where a Telegram DM maps to
    ``tg-<chat_id>``.  Existing entries are moved on first load; the legacy key
    is dropped on the next save.  Safe to delete once no deployment holds it.

    Hand-edited settings files reach this on every inbound message, so malformed shapes
    are passed through to Pydantic for a field-level error rather than raising here.
    """
    if not isinstance(data, dict):
        return data
    telegram = data.get("telegram")
    chats = telegram.get("chats") if isinstance(telegram, dict) else None
    if not isinstance(chats, dict) or not chats:
        return data
    conversations = data.setdefault("conversations", {})
    if not isinstance(conversations, dict):
        return data
    for chat_id, chat in chats.items():
        agent = chat.get("agent") if isinstance(chat, dict) else None
        if agent:
            conversations.setdefault(f"tg-{chat_id}", {"agent": agent})
    return data
