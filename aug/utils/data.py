"""Utilities for reading files from the /app/data volume."""

from pathlib import Path

DATA_DIR = Path("/app/data")
MEMORY_DIR = DATA_DIR / "memory"


def read_data_file(name: str) -> str:
    """Read a file from DATA_DIR by name. Returns empty string if missing."""
    try:
        return (DATA_DIR / name).read_text().strip()
    except FileNotFoundError:
        return ""


def write_data_file(name: str, content: str) -> None:
    """Write content to a file in DATA_DIR by name, creating the directory if needed."""
    path = DATA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
