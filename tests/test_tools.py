"""Unit tests for individual tools."""

from pathlib import Path
from unittest.mock import patch

from aug.core.tools.note import note
from aug.core.tools.run_bash import _check_blacklist

# ---------------------------------------------------------------------------
# note tool
# ---------------------------------------------------------------------------


def test_note_creates_file(tmp_path: Path) -> None:
    with patch("aug.core.tools.note.MEMORY_DIR", tmp_path):
        result = note.invoke({"content": "user prefers dark mode"})

    assert result == "Noted."
    notes = (tmp_path / "notes.md").read_text()
    assert "user prefers dark mode" in notes


def test_note_appends(tmp_path: Path) -> None:
    with patch("aug.core.tools.note.MEMORY_DIR", tmp_path):
        note.invoke({"content": "first note"})
        note.invoke({"content": "second note"})

    notes = (tmp_path / "notes.md").read_text()
    assert "first note" in notes
    assert "second note" in notes


def test_note_includes_timestamp(tmp_path: Path) -> None:
    with patch("aug.core.tools.note.MEMORY_DIR", tmp_path):
        note.invoke({"content": "timestamped"})

    notes = (tmp_path / "notes.md").read_text()
    # Timestamp format: [2024-01-01 00:00:00 UTC]
    assert "UTC]" in notes


# ---------------------------------------------------------------------------
# run_bash blacklist
# ---------------------------------------------------------------------------


def test_blacklist_allows_clean_command() -> None:
    with patch("aug.core.tools.run_bash.get_setting", return_value=["rm -rf"]):
        assert _check_blacklist("ls -la") is None


def test_blacklist_blocks_matching_command() -> None:
    with patch("aug.core.tools.run_bash.get_setting", return_value=[r"rm\s+-rf"]):
        result = _check_blacklist("rm -rf /")
        assert result is not None
        assert "blacklist" in result.lower()


def test_blacklist_empty_by_default() -> None:
    with patch("aug.core.tools.run_bash.get_setting", return_value=[]):
        assert _check_blacklist("anything") is None


def test_blacklist_uses_regex() -> None:
    with patch("aug.core.tools.run_bash.get_setting", return_value=[r"DROP\s+TABLE"]):
        assert _check_blacklist("DROP TABLE users") is not None
        assert _check_blacklist("drop table users") is None  # case-sensitive
