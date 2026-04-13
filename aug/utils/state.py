"""Runtime state persisted across restarts.

Stores values written by the app itself (counters, scheduler timestamps).
Not for user-facing configuration — use aug/utils/user_settings.py for that.

API mirrors user_settings.py intentionally.
"""

import json
from typing import Any

from aug.utils.data import read_data_file, write_data_file

_STATE_FILE = "state.json"


def get_state(*path: str, default: Any = None) -> Any:
    """Read a value at an arbitrary nested path."""
    node = _load()
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def set_state(*path: str, value: Any) -> None:
    """Write a value at an arbitrary nested path, creating intermediate dicts as needed."""
    data = _load()
    node = data
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value
    write_data_file(_STATE_FILE, json.dumps(data, indent=2))


def _load() -> dict:
    raw = read_data_file(_STATE_FILE)
    if not raw:
        return {}
    return json.loads(raw)
