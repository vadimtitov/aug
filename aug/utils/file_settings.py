"""Typed settings backed by data/settings.json.

All user-facing configuration lives here. Access via:

    from aug.utils.file_settings import load_settings, save_settings

    s = load_settings()
    agent = s.telegram.chats.get(chat_id, TelegramChatSettings()).agent

    s = load_settings()
    s.telegram.chats[chat_id] = TelegramChatSettings(agent="v2")
    save_settings(s)
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from aug.utils.data import read_data_file, write_data_file

_SETTINGS_FILE = "settings.json"


class TelegramChatSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str = "default"


class TelegramSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chats: dict[str, TelegramChatSettings] = {}


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

    telegram: TelegramSettings = TelegramSettings()
    consolidation: ConsolidationSettings = ConsolidationSettings()
    tools: ToolSettings = ToolSettings()
    reflexes: ReflexSettings = ReflexSettings()


def load_settings() -> AppSettings:
    """Load settings from data/settings.json, filling in defaults for any missing fields."""
    raw = read_data_file(_SETTINGS_FILE)
    if not raw:
        return AppSettings()
    return AppSettings.model_validate_json(raw)


def save_settings(settings: AppSettings) -> None:
    """Persist settings to data/settings.json."""
    write_data_file(_SETTINGS_FILE, json.dumps(settings.model_dump(), indent=2))
