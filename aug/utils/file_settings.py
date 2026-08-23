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

import json

from pydantic import BaseModel, ConfigDict

from aug.utils.data import read_data_file, write_data_file

_SETTINGS_FILE = "settings.json"


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


def load_settings() -> AppSettings:
    """Load settings from data/settings.json, filling in defaults for any missing fields."""
    raw = read_data_file(_SETTINGS_FILE)
    if not raw:
        return AppSettings()
    return AppSettings.model_validate(_migrate(json.loads(raw)))


def save_settings(settings: AppSettings) -> None:
    """Persist settings to data/settings.json."""
    write_data_file(_SETTINGS_FILE, json.dumps(settings.model_dump(), indent=2))


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
