"""Utilities for reading files from the /app/data volume."""

import os
import tempfile
from pathlib import Path

DATA_DIR = Path("/app/data")
MEMORY_DIR = DATA_DIR / "memory"
UPLOADS_DIR = DATA_DIR / "uploads"


def read_data_file(name: str) -> str:
    """Read a file from DATA_DIR by name. Returns empty string if missing."""
    try:
        return (DATA_DIR / name).read_text().strip()
    except FileNotFoundError:
        return ""


def write_data_file(name: str, content: str) -> None:
    """Write content to a file in DATA_DIR atomically.

    Writes to a sibling temp file first, then uses os.replace() which is an
    atomic operation on POSIX systems. This prevents partial reads if two
    writes race, and avoids leaving a corrupt file on process crash.
    """
    path = DATA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
