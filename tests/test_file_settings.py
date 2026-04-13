"""Tests for aug/utils/file_settings.py — Pydantic-typed settings models."""

import json
from unittest.mock import patch

from aug.utils.file_settings import (
    ApprovalRule,
    AppSettings,
    SshTarget,
    TelegramChatSettings,
    load_settings,
    save_settings,
)

# ---------------------------------------------------------------------------
# load_settings — defaults
# ---------------------------------------------------------------------------


def test_load_returns_defaults_when_file_empty():
    with patch("aug.utils.file_settings.read_data_file", return_value=""):
        s = load_settings()
    assert s.telegram.chats == {}
    assert s.consolidation.model == "gpt-5.1"
    assert s.tools.approvals == []
    assert s.tools.ssh.targets == []
    assert s.tools.bash.blacklist == []
    assert s.reflexes.homeassistant.entity_label == "aug"
    assert s.tools.browser.model == "gpt-5.1"
    assert s.tools.image_gen.model == "gpt-image-1.5"
    assert s.tools.ssh.max_download_bytes == 1_073_741_824


def test_load_ignores_unknown_top_level_keys():
    raw = json.dumps({"unknown_section": {"foo": "bar"}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert isinstance(s, AppSettings)


# ---------------------------------------------------------------------------
# load_settings — telegram
# ---------------------------------------------------------------------------


def test_load_telegram_chat_agent():
    raw = json.dumps({"telegram": {"chats": {"123": {"agent": "v2_claude"}}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.telegram.chats["123"].agent == "v2_claude"


def test_load_telegram_missing_chat_defaults_to_default_agent():
    raw = json.dumps({"telegram": {"chats": {}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.telegram.chats.get("999", TelegramChatSettings()).agent == "default"


def test_load_telegram_chat_ignores_unknown_fields():
    raw = json.dumps({"telegram": {"chats": {"1": {"agent": "v1", "unknown": "x"}}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.telegram.chats["1"].agent == "v1"


# ---------------------------------------------------------------------------
# load_settings — SSH targets
# ---------------------------------------------------------------------------


def test_load_ssh_targets_full():
    raw = json.dumps(
        {
            "tools": {
                "ssh": {
                    "targets": [
                        {
                            "name": "home",
                            "host": "192.168.1.1",
                            "port": 2222,
                            "user": "vadim",
                            "key_path": "/keys/home.pem",
                            "known_hosts": "/keys/home.known_hosts",
                        }
                    ]
                }
            }
        }
    )
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    t = s.tools.ssh.targets[0]
    assert t.name == "home"
    assert t.host == "192.168.1.1"
    assert t.port == 2222
    assert t.user == "vadim"
    assert t.key_path == "/keys/home.pem"
    assert t.known_hosts == "/keys/home.known_hosts"


def test_load_ssh_target_defaults():
    raw = json.dumps(
        {
            "tools": {
                "ssh": {
                    "targets": [{"name": "x", "host": "1.2.3.4", "user": "u", "key_path": "/k"}]
                }
            }
        }
    )
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    t = s.tools.ssh.targets[0]
    assert t.port == 22
    assert t.verify_host is True
    assert t.known_hosts == ""


def test_load_ssh_max_download_bytes():
    raw = json.dumps({"tools": {"ssh": {"max_download_bytes": 50_000_000}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.tools.ssh.max_download_bytes == 50_000_000


# ---------------------------------------------------------------------------
# load_settings — approval rules
# ---------------------------------------------------------------------------


def test_load_approval_rules():
    raw = json.dumps(
        {"tools": {"approvals": [{"tool": "run_ssh", "target": "home", "pattern": "df.*"}]}}
    )
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    r = s.tools.approvals[0]
    assert r.tool == "run_ssh"
    assert r.target == "home"
    assert r.pattern == "df.*"


def test_load_approval_rule_defaults_to_wildcard():
    raw = json.dumps({"tools": {"approvals": [{"pattern": "uptime"}]}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    r = s.tools.approvals[0]
    assert r.tool == "*"
    assert r.target == "*"


# ---------------------------------------------------------------------------
# load_settings — tool models
# ---------------------------------------------------------------------------


def test_load_browser_model():
    raw = json.dumps({"tools": {"browser": {"model": "claude-3-5"}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.tools.browser.model == "claude-3-5"


def test_load_image_gen_model():
    raw = json.dumps({"tools": {"image_gen": {"model": "dall-e-3"}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.tools.image_gen.model == "dall-e-3"


def test_load_bash_blacklist():
    raw = json.dumps({"tools": {"bash": {"blacklist": ["rm -rf", "sudo"]}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.tools.bash.blacklist == ["rm -rf", "sudo"]


def test_load_consolidation_model():
    raw = json.dumps({"consolidation": {"model": "claude-opus-4"}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.consolidation.model == "claude-opus-4"


def test_load_reflexes_ha_entity_label():
    raw = json.dumps({"reflexes": {"homeassistant": {"entity_label": "mybot"}}})
    with patch("aug.utils.file_settings.read_data_file", return_value=raw):
        s = load_settings()
    assert s.reflexes.homeassistant.entity_label == "mybot"


# ---------------------------------------------------------------------------
# save_settings — round-trip
# ---------------------------------------------------------------------------


def test_save_settings_round_trips_telegram():
    written: list[str] = []
    s = AppSettings()
    s.telegram.chats["42"] = TelegramChatSettings(agent="v3")

    with patch(
        "aug.utils.file_settings.write_data_file", side_effect=lambda _f, d: written.append(d)
    ):
        save_settings(s)

    loaded = AppSettings.model_validate_json(written[0])
    assert loaded.telegram.chats["42"].agent == "v3"


def test_save_settings_round_trips_approvals():
    written: list[str] = []
    s = AppSettings()
    s.tools.approvals.append(ApprovalRule(tool="run_ssh", target="box", pattern="ls"))

    with patch(
        "aug.utils.file_settings.write_data_file", side_effect=lambda _f, d: written.append(d)
    ):
        save_settings(s)

    loaded = AppSettings.model_validate_json(written[0])
    assert loaded.tools.approvals[0].tool == "run_ssh"
    assert loaded.tools.approvals[0].pattern == "ls"


def test_save_settings_round_trips_ssh_targets():
    written: list[str] = []
    s = AppSettings()
    s.tools.ssh.targets.append(
        SshTarget(name="srv", host="10.0.0.1", user="admin", key_path="/keys/srv.pem")
    )

    with patch(
        "aug.utils.file_settings.write_data_file", side_effect=lambda _f, d: written.append(d)
    ):
        save_settings(s)

    loaded = AppSettings.model_validate_json(written[0])
    assert loaded.tools.ssh.targets[0].name == "srv"
    assert loaded.tools.ssh.targets[0].host == "10.0.0.1"


def test_save_writes_to_settings_file():
    filename_used: list[str] = []
    s = AppSettings()

    with patch(
        "aug.utils.file_settings.write_data_file",
        side_effect=lambda f, _d: filename_used.append(f),
    ):
        save_settings(s)

    assert filename_used == ["settings.json"]
