"""Unit tests for memory file initialisation."""

from pathlib import Path
from unittest.mock import patch

from aug.core.memory import append_note, init_memory_files


def test_init_creates_files(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    assert (memory_dir / "self.md").exists()
    assert (memory_dir / "user.md").exists()
    assert (memory_dir / "context.md").exists()
    assert (memory_dir / "memory.md").exists()
    assert (memory_dir / "reflections.md").exists()
    assert (memory_dir / "notes.md").exists()


def test_init_self_md_has_content(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    content = (memory_dir / "self.md").read_text()
    assert len(content.strip()) > 0


def test_init_memory_md_has_sections(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    assert "Patterns" in (memory_dir / "memory.md").read_text()
    assert "Significant moments" in (memory_dir / "memory.md").read_text()
    assert "Present" in (memory_dir / "context.md").read_text()
    assert "Recent" in (memory_dir / "context.md").read_text()


def test_init_does_not_overwrite_existing(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "self.md").write_text("my custom identity")

    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    assert (memory_dir / "self.md").read_text() == "my custom identity"


# ---------------------------------------------------------------------------
# append_note ring buffer
# ---------------------------------------------------------------------------


def test_append_note_writes_entry(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        append_note("hello world")

    content = (tmp_path / "notes.md").read_text()
    assert "hello world" in content
    assert "UTC]" in content


def test_append_note_preserves_existing(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        append_note("first")
        append_note("second")

    content = (tmp_path / "notes.md").read_text()
    assert "first" in content
    assert "second" in content


def test_append_note_ring_buffer(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        with patch("aug.core.memory._NOTES_MAX_LINES", 5):
            for i in range(7):
                append_note(f"note {i}")

    lines = [ln for ln in (tmp_path / "notes.md").read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    assert "note 2" in lines[0]
    assert "note 6" in lines[-1]
