"""Unit tests for individual tools."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from aug.core.tools.note import note
from aug.core.tools.run_bash import _check_blacklist
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
# run_bash — non-interactive execution
# ---------------------------------------------------------------------------


def _run_bash(command: str):
    """Invoke the run_bash tool with the blacklist disabled."""
    from aug.core.tools.run_bash import run_bash

    with patch("aug.core.tools.run_bash.load_settings", return_value=_bash_settings([])):
        return run_bash.invoke({"command": command})


# Runs run_bash in a child process that has a real stdin carrying data. In-process this
# cannot be tested: pytest already replaces stdin with a null reader, so an inherited
# stdin and a DEVNULL stdin look identical and the assertion passes either way.
_STDIN_LEAK_SCRIPT = """
import os, subprocess as sp
os.environ.setdefault("API_KEY", "t")
os.environ.setdefault("LLM_API_KEY", "t")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:4000")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://t:t@localhost:5432/t")
from unittest.mock import patch
from aug.core.tools.run_bash import run_bash
from aug.utils.file_settings import AppSettings

real = sp.run
with patch("aug.core.tools.run_bash.subprocess.run", side_effect=lambda c, **k: real(c[3:], **k)), \
     patch("aug.core.tools.run_bash.load_settings", return_value=AppSettings()):
    print(run_bash.invoke({"command": "read -r l && echo GOT:$l || echo EOF-ON-STDIN"}))
"""


def test_run_bash_does_not_inherit_the_servers_stdin():
    """A command that reads stdin must get EOF, never the parent process's input.

    The parent here is handed real data on stdin. Without stdin=DEVNULL the child bash
    inherits that pipe and reads it — in production that pipe is the server's own stdin,
    which is how `hushed` ended up prompting into a non-TTY and looping on ENOTTY.
    """
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _STDIN_LEAK_SCRIPT],
        input="leaked-from-parent\n",
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert result.returncode == 0, result.stderr
    assert "EOF-ON-STDIN" in result.stdout
    assert "leaked-from-parent" not in result.stdout


def test_run_bash_passes_devnull_and_noninteractive_env():
    with patch("aug.core.tools.run_bash.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        _run_bash("echo hi")

    kwargs = mock_run.call_args.kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert kwargs["env"]["DEBIAN_FRONTEND"] == "noninteractive"
    assert "PATH" in kwargs["env"]  # inherits the real environment (hushed secrets)


def test_run_bash_reports_prompt_failure_unambiguously():
    """The ENOTTY loop must come back as an explicit failure, not neutral output.

    Returning the raw 'Enter value:' spam with no verdict invites the agent to report
    success to the user.
    """
    with patch("aug.core.tools.run_bash.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Enter value: \n" * 3,
            stderr="Error: read password: inappropriate ioctl for device\n",
        )
        out = _run_bash("hushed set FOO")

    assert "did NOT complete" in out
    assert "non-interactive" in out


def test_run_bash_timeout_is_a_clean_error():
    import subprocess as sp

    with patch("aug.core.tools.run_bash.subprocess.run", side_effect=sp.TimeoutExpired("x", 60)):
        out = _run_bash("sleep 999")

    assert "did NOT complete" in out
    assert "timed out" in out


def test_run_bash_missing_hushed_binary():
    with patch("aug.core.tools.run_bash.subprocess.run", side_effect=FileNotFoundError):
        out = _run_bash("echo hi")

    assert "did NOT run" in out
    assert "hushed" in out


def test_run_bash_nonzero_exit_is_labelled_as_failure():
    with patch("aug.core.tools.run_bash.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="no such file")
        out = _run_bash("cat /nope")

    assert "failed (exit 2)" in out
    assert "no such file" in out
