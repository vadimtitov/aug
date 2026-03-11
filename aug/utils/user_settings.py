"""User settings stored in /app/data/settings.json.

Schema:
  {
    "<namespace>": {            # e.g. "telegram"
      "<sub-key>": {            # e.g. "chats"
        "<entity-id>": {        # e.g. telegram chat ID
          "<setting>": <value>  # e.g. "agent": "default"
        }
      }
    }
  }

Example — telegram per-chat settings:
  get_setting("telegram", "chats", str(chat_id), "agent", default="default")
  set_setting("telegram", "chats", str(chat_id), "agent", value="v1_claude")
"""

import json
from typing import Any

from aug.utils.data import read_data_file, write_data_file

_SETTINGS_FILE = "settings.json"


def get_setting(*path: str, default: Any = None) -> Any:
    """Read a value at an arbitrary nested path."""
    node = _load()
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def set_setting(*path: str, value: Any) -> None:
    """Write a value at an arbitrary nested path, creating intermediate dicts as needed."""
    data = _load()
    node = data
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value
    write_data_file(_SETTINGS_FILE, json.dumps(data, indent=2))


def _load() -> dict:
    raw = read_data_file(_SETTINGS_FILE)
    if not raw:
        return {}
    return json.loads(raw)
