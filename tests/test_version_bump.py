"""scripts/check_version_bump.py: the CI gate that every change raises the version."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

import aug

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_version_bump.py"


@pytest.mark.parametrize(
    ("old", "new"),
    [("1.0.0", "1.0.1"), ("1.0.9", "1.1.0"), ("1.9.9", "2.0.0"), ("1.0.9", "1.0.10")],
)
def test_a_higher_version_passes(tmp_path, old, new):
    result = _check(tmp_path, _init(old), _init(new))
    assert result.returncode == 0, result.stderr
    assert f"{old} -> {new}" in result.stdout


@pytest.mark.parametrize(
    ("old", "new"), [("1.0.0", "1.0.0"), ("1.2.0", "1.1.9"), ("1.0.10", "1.0.9")]
)
def test_an_unchanged_or_lower_version_fails(tmp_path, old, new):
    result = _check(tmp_path, _init(old), _init(new))
    assert result.returncode == 1
    assert "Bump __version__" in result.stderr


@pytest.mark.parametrize("version", ["1.0", "1.0.0rc1", "v1.0.0"])
def test_a_version_that_is_not_major_minor_patch_fails(tmp_path, version):
    result = _check(tmp_path, _init("1.0.0"), _init(version))
    assert result.returncode == 1


def test_a_target_branch_without_a_version_passes(tmp_path):
    result = _check(tmp_path, '"""AUG."""\n', _init("1.0.0"))
    assert result.returncode == 0, result.stderr


def test_the_installed_package_carries_the_source_version():
    assert importlib.metadata.version("aug") == aug.__version__


def _check(tmp_path: Path, base: str, head: str) -> subprocess.CompletedProcess[str]:
    base_file, head_file = tmp_path / "base.py", tmp_path / "head.py"
    base_file.write_text(base)
    head_file.write_text(head)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_file), str(head_file)],
        capture_output=True,
        text=True,
    )


def _init(version: str) -> str:
    return f'"""AUG."""\n\n__version__ = "{version}"\n'
