"""Unit tests for individual tools."""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aug.core.tools.note import note
from aug.core.tools.run_bash import _check_blacklist, run_bash
from aug.utils.file_settings import AppSettings, BashToolSettings, ToolSettings


def _bash_settings(blacklist: list[str]) -> AppSettings:
    return AppSettings(tools=ToolSettings(bash=BashToolSettings(blacklist=blacklist)))


# ---------------------------------------------------------------------------
# note tool
# ---------------------------------------------------------------------------


def test_note_creates_file(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        result = note.invoke({"content": "user prefers dark mode"})

    assert result == "Noted."
    notes = (tmp_path / "notes.md").read_text()
    assert "user prefers dark mode" in notes


def test_note_appends(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        note.invoke({"content": "first note"})
        note.invoke({"content": "second note"})

    notes = (tmp_path / "notes.md").read_text()
    assert "first note" in notes
    assert "second note" in notes


def test_note_includes_timestamp(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        note.invoke({"content": "timestamped"})

    notes = (tmp_path / "notes.md").read_text()
    # Timestamp format: [2024-01-01 00:00:00 UTC]
    assert "UTC]" in notes


def test_note_description_prohibits_credentials() -> None:
    desc = note.description or ""
    assert "password" in desc.lower() or "credential" in desc.lower()


def test_note_description_no_skill_update_instruction() -> None:
    desc = note.description or ""
    assert "update that skill" not in desc.lower()


# ---------------------------------------------------------------------------
# run_bash blacklist
# ---------------------------------------------------------------------------


def test_blacklist_allows_clean_command() -> None:
    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings(["rm -rf"])):
        assert _check_blacklist("ls -la") is None


def test_blacklist_blocks_matching_command() -> None:
    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings([r"rm\s+-rf"])):
        result = _check_blacklist("rm -rf /")
        assert result is not None
        assert "blacklist" in result.lower()


def test_blacklist_empty_by_default() -> None:
    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings([])):
        assert _check_blacklist("anything") is None


def test_blacklist_uses_regex() -> None:
    pattern = [r"DROP\s+TABLE"]
    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings(pattern)):
        assert _check_blacklist("DROP TABLE users") is not None
        assert _check_blacklist("drop table users") is None  # case-sensitive


# ---------------------------------------------------------------------------
# run_bash execution and /stop
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_hushed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `hushed` on PATH that just runs the command (`hushed run -- bash -c CMD`)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "hushed"
    script.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings([])):
        yield tmp_path


async def test_run_bash_returns_stdout_and_stderr(fake_hushed: Path) -> None:
    result = await run_bash.ainvoke({"command": "echo out; echo err >&2; exit 3"})
    assert "out" in result and "err" in result


async def test_run_bash_returns_without_waiting_for_a_background_job(fake_hushed: Path) -> None:
    started = time.monotonic()
    result = await run_bash.ainvoke({"command": "sleep 5 & echo started"})
    assert result == "started"
    assert time.monotonic() - started < 2


async def test_run_bash_keeps_the_exit_code_and_both_streams(fake_hushed: Path) -> None:
    result = await run_bash.ainvoke({"command": "echo out; echo err >&2; exec false"})
    assert result == "out\nerr"


async def test_run_bash_cancel_kills_the_command_and_its_children(fake_hushed: Path) -> None:
    pid_file = fake_hushed / "child.pid"
    task = asyncio.create_task(
        run_bash.ainvoke({"command": f"sleep 30 & echo $! > {pid_file}; sleep 30"})
    )
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.01)
    child = int(pid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(200):
        if not _alive(child):
            break
        await asyncio.sleep(0.01)
    assert not _alive(child)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
