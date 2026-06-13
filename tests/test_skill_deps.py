"""Unit tests for skill dependencies (aug/core/skill_deps.py)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from aug.core.skill_deps import (
    _is_pep723,
    find_pep723_scripts,
    inject_dependencies,
    read_pep723_deps,
    resolve_dependencies,
    warm_skill_dir,
)

_PEP723_SCRIPT = """\
# /// script
# dependencies = ["beautifulsoup4>=4.12,<5"]
# ///

print("hi")
"""

_PLAIN_SCRIPT = "import os\nprint(os.getcwd())\n"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_is_pep723_detects_marker(tmp_path):
    assert _is_pep723(_write(tmp_path / "a.py", _PEP723_SCRIPT)) is True


def test_is_pep723_ignores_plain_script(tmp_path):
    assert _is_pep723(_write(tmp_path / "b.py", _PLAIN_SCRIPT)) is False


def test_find_pep723_scripts_recurses_and_filters(tmp_path):
    _write(tmp_path / "skill-a" / "scripts" / "dep.py", _PEP723_SCRIPT)
    _write(tmp_path / "skill-a" / "scripts" / "plain.py", _PLAIN_SCRIPT)
    _write(tmp_path / "skill-b" / "SKILL.md", "---\nname: skill-b\n---\n")
    _write(tmp_path / "skill-b" / "run.sh", "echo hi")  # non-python ignored

    found = find_pep723_scripts(tmp_path)
    assert found == [tmp_path / "skill-a" / "scripts" / "dep.py"]


def test_find_pep723_scripts_missing_dir_returns_empty(tmp_path):
    assert find_pep723_scripts(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# inject_dependencies / read_pep723_deps
# ---------------------------------------------------------------------------


def test_inject_dependencies_prepends_block_and_is_readable(tmp_path):
    out = inject_dependencies("import httpx\nprint('hi')\n", ["httpx>=0.27", "rich"])
    assert out.startswith("# /// script\n")
    assert "import httpx" in out
    script = _write(tmp_path / "s.py", out)
    assert read_pep723_deps(script) == ["httpx>=0.27", "rich"]


def test_inject_dependencies_replaces_existing_block(tmp_path):
    out = inject_dependencies(_PEP723_SCRIPT, ["requests==2.32"])
    # Old dependency is gone, new one present, body preserved, only one block.
    assert "beautifulsoup4" not in out
    assert out.count("# /// script") == 1
    assert 'print("hi")' in out
    script = _write(tmp_path / "s.py", out)
    assert read_pep723_deps(script) == ["requests==2.32"]


def test_inject_dependencies_preserves_shebang(tmp_path):
    out = inject_dependencies("#!/usr/bin/env python3\nimport x\n", ["x"])
    assert out.startswith("#!/usr/bin/env python3\n# /// script\n")
    script = _write(tmp_path / "s.py", out)
    assert read_pep723_deps(script) == ["x"]


def test_inject_dependencies_empty_removes_block():
    out = inject_dependencies(_PEP723_SCRIPT, [])
    assert "# /// script" not in out
    assert 'print("hi")' in out


def test_read_pep723_deps_returns_empty_for_plain_script(tmp_path):
    assert read_pep723_deps(_write(tmp_path / "p.py", _PLAIN_SCRIPT)) == []


def test_inject_dependencies_handles_crlf_without_double_block(tmp_path):
    # CRLF content with an existing block must be re-injected as a single block, not two —
    # uv rejects a script with two '# /// script' blocks.
    crlf = inject_dependencies("import x\n", ["a"]).replace("\n", "\r\n")
    out = inject_dependencies(crlf, ["b"])
    assert out.count("# /// script") == 1
    script = _write(tmp_path / "s.py", out)
    assert read_pep723_deps(script) == ["b"]


def test_read_pep723_deps_handles_crlf(tmp_path):
    crlf_script = _PEP723_SCRIPT.replace("\n", "\r\n")
    assert read_pep723_deps(_write(tmp_path / "c.py", crlf_script)) == ["beautifulsoup4>=4.12,<5"]


# ---------------------------------------------------------------------------
# resolve_dependencies
# ---------------------------------------------------------------------------


def test_resolve_dependencies_success(tmp_path):
    script = _write(tmp_path / "s.py", _PEP723_SCRIPT)
    with patch("aug.core.skill_deps.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = resolve_dependencies(script)
    assert result.ok is True
    mock_run.assert_called_once()
    assert tuple(mock_run.call_args.args[0]) == ("uv", "sync", "--script", str(script))


def test_resolve_dependencies_reports_failure_detail(tmp_path):
    script = _write(tmp_path / "s.py", _PEP723_SCRIPT)
    with patch("aug.core.skill_deps.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No matching distribution for 'nope'"
        )
        result = resolve_dependencies(script)
    assert result.ok is False
    assert "No matching distribution" in result.detail


def test_resolve_dependencies_handles_missing_uv(tmp_path):
    script = _write(tmp_path / "s.py", _PEP723_SCRIPT)
    with patch("aug.core.skill_deps.subprocess.run", side_effect=FileNotFoundError):
        result = resolve_dependencies(script)
    assert result.ok is False
    assert "uv is not available" in result.detail


def test_resolve_dependencies_handles_timeout(tmp_path):
    script = _write(tmp_path / "s.py", _PEP723_SCRIPT)
    with patch(
        "aug.core.skill_deps.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=600),
    ):
        result = resolve_dependencies(script)
    assert result.ok is False
    assert "timed out" in result.detail


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def test_warm_skill_dir_runs_uv_sync_for_each_script(tmp_path):
    s1 = _write(tmp_path / "skill" / "one.py", _PEP723_SCRIPT)
    s2 = _write(tmp_path / "skill" / "two.py", _PEP723_SCRIPT)

    with patch("aug.core.skill_deps.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        warm_skill_dir(tmp_path)

    called = {tuple(call.args[0]) for call in mock_run.call_args_list}
    assert ("uv", "sync", "--script", str(s1)) in called
    assert ("uv", "sync", "--script", str(s2)) in called
    assert mock_run.call_count == 2


def test_warm_skill_dir_no_scripts_does_not_invoke_uv(tmp_path):
    _write(tmp_path / "skill" / "plain.py", _PLAIN_SCRIPT)
    with patch("aug.core.skill_deps.subprocess.run") as mock_run:
        warm_skill_dir(tmp_path)
    mock_run.assert_not_called()


def test_warm_skill_dir_survives_uv_failure(tmp_path):
    _write(tmp_path / "skill" / "dep.py", _PEP723_SCRIPT)
    with patch("aug.core.skill_deps.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="resolution failed")
        warm_skill_dir(tmp_path)  # must not raise
    mock_run.assert_called_once()
