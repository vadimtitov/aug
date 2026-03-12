"""Unit tests for consolidation utilities."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

from aug.core.consolidation import _extract, _iso_date, _iso_week, _read, _write

# ---------------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------------


def test_extract_found() -> None:
    text = "<memory>some content here</memory>"
    assert _extract("memory", text) == "some content here"


def test_extract_multiline() -> None:
    text = "<user>\nline one\nline two\n</user>"
    assert _extract("user", text) == "line one\nline two"


def test_extract_missing_tag() -> None:
    assert _extract("memory", "no tags here") is None


def test_extract_trims_whitespace() -> None:
    assert _extract("self", "<self>  trimmed  </self>") == "trimmed"


# ---------------------------------------------------------------------------
# _iso_date / _iso_week
# ---------------------------------------------------------------------------


def test_iso_date_parses() -> None:
    assert _iso_date("2024-03-15T03:00:00+00:00") == date(2024, 3, 15)


def test_iso_date_none() -> None:
    assert _iso_date(None) is None


def test_iso_week_parses() -> None:
    # 2024-03-15 is week 11
    assert _iso_week("2024-03-15T00:00:00+00:00") == 11


def test_iso_week_none() -> None:
    assert _iso_week(None) is None


# ---------------------------------------------------------------------------
# _read / _write
# ---------------------------------------------------------------------------


def test_read_returns_empty_for_missing_file(tmp_path: Path) -> None:
    with patch("aug.core.consolidation.MEMORY_DIR", tmp_path):
        assert _read("nonexistent.md") == ""


def test_read_returns_stripped_content(tmp_path: Path) -> None:
    (tmp_path / "memory.md").write_text("  hello  \n\n")
    with patch("aug.core.consolidation.MEMORY_DIR", tmp_path):
        assert _read("memory.md") == "hello"


def test_write_creates_file(tmp_path: Path) -> None:
    with patch("aug.core.consolidation.MEMORY_DIR", tmp_path):
        _write("memory.md", "new content")
    assert (tmp_path / "memory.md").read_text() == "new content\n"


def test_write_overwrites(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("old")
    with patch("aug.core.consolidation.MEMORY_DIR", tmp_path):
        _write("notes.md", "")
    assert (tmp_path / "notes.md").read_text() == "\n"
