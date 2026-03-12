"""Unit tests for memory file initialisation."""

from pathlib import Path
from unittest.mock import patch

from aug.core.memory import init_memory_files


def test_init_creates_files(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    assert (memory_dir / "self.md").exists()
    assert (memory_dir / "user.md").exists()
    assert (memory_dir / "memory.md").exists()
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

    content = (memory_dir / "memory.md").read_text()
    sections = ("Present", "Recent", "Patterns", "Significant moments", "Reflections", "Longer arc")
    for section in sections:
        assert section in content


def test_init_does_not_overwrite_existing(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "self.md").write_text("my custom identity")

    with patch("aug.core.memory.MEMORY_DIR", memory_dir):
        init_memory_files()

    assert (memory_dir / "self.md").read_text() == "my custom identity"
