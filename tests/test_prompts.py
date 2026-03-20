"""Unit tests for build_system_prompt."""

from pathlib import Path
from unittest.mock import patch

from aug.core.prompts import build_system_prompt
from aug.core.state import AgentState


def _make_state(interface: str = "") -> AgentState:
    return AgentState(interface=interface)


def _build(tmp_path: Path, interface: str = "", files: dict[str, str] | None = None) -> str:
    files = files or {}
    with patch("aug.core.prompts.MEMORY_DIR", tmp_path):
        for name, content in files.items():
            (tmp_path / name).write_text(content)
        return build_system_prompt(_make_state(interface))


# ---------------------------------------------------------------------------
# Section presence
# ---------------------------------------------------------------------------


def test_self_section_present(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"self.md": "I am AUG."})
    assert "<self>" in prompt
    assert "I am AUG." in prompt


def test_approach_section_present(tmp_path: Path) -> None:
    prompt = _build(tmp_path)
    assert "<approach>" in prompt


def test_user_section_present(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"user.md": "Name: V"})
    assert "<user>" in prompt
    assert "Name: V" in prompt


def test_context_section_present(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"context.md": "## Present\nworking on infra"})
    assert "<context>" in prompt
    assert "working on infra" in prompt


def test_memory_section_present(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"memory.md": "## Patterns\nprefers brevity"})
    assert "<memory>" in prompt
    assert "prefers brevity" in prompt


# ---------------------------------------------------------------------------
# Removed sections
# ---------------------------------------------------------------------------


def test_no_structure_section(tmp_path: Path) -> None:
    prompt = _build(tmp_path)
    assert "<structure>" not in prompt


def test_no_memory_system_section(tmp_path: Path) -> None:
    prompt = _build(tmp_path)
    assert "<memory-system>" not in prompt


# ---------------------------------------------------------------------------
# Skills section — conditional
# ---------------------------------------------------------------------------


def test_skills_absent_when_empty(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"skills.md": ""})
    assert "<skills>" not in prompt


def test_skills_absent_when_missing(tmp_path: Path) -> None:
    prompt = _build(tmp_path)
    assert "<skills>" not in prompt


def test_skills_present_when_non_empty(tmp_path: Path) -> None:
    prompt = _build(tmp_path, files={"skills.md": "Home Assistant: HA_URL + HASS_TOKEN"})
    assert "<skills>" in prompt
    assert "Home Assistant" in prompt


# ---------------------------------------------------------------------------
# Interface sections — conditional
# ---------------------------------------------------------------------------


def test_interface_section_present_for_telegram(tmp_path: Path) -> None:
    prompt = _build(tmp_path, interface="telegram")
    assert "<interface>" in prompt
    assert "<response_format>" in prompt


def test_interface_section_absent_for_unknown(tmp_path: Path) -> None:
    prompt = _build(tmp_path, interface="unknown")
    assert "<interface>" not in prompt
    assert "<response_format>" not in prompt
