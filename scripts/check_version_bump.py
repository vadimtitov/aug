"""Fail unless a change raises AUG's version.

Usage: check_version_bump.py BASE_INIT HEAD_INIT

Both arguments are copies of ``aug/__init__.py``: the one on the branch the change
merges into, and the changed one.  The version must be ``MAJOR.MINOR.PATCH`` and
higher in HEAD_INIT.
"""

import re
import sys
from pathlib import Path

_VERSION = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def main(base: Path, head: Path) -> int:
    old, new = _read(base), _read(head)
    if new is None:
        print(f'{head}: no __version__ = "MAJOR.MINOR.PATCH" line', file=sys.stderr)
        return 1
    if old is not None and new <= old:
        print(
            f"Version {_show(new)} is not higher than {_show(old)} on the target branch. "
            "Bump __version__ in aug/__init__.py: MAJOR for breaking changes, "
            "MINOR for features, PATCH for fixes.",
            file=sys.stderr,
        )
        return 1
    print(f"Version {_show(old) if old else '(none)'} -> {_show(new)}")
    return 0


def _read(path: Path) -> tuple[int, int, int] | None:
    match = _VERSION.search(path.read_text())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _show(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
