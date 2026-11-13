"""Paths no tool may read or send: secret material and AUG's own settings.

``run_bash`` refuses commands that name them; ``respond_with_file`` refuses to send
anything under them.  The user's own blacklist patterns (settings.json →
tools.bash.blacklist) apply to both as well.
"""

import logging
import re
from pathlib import Path

from aug.utils.data import DATA_DIR
from aug.utils.file_settings import load_settings

logger = logging.getLogger(__name__)

# SSH private keys, the hushed secret store (its age key sits next to it), and
# settings.json, which holds the saved approval rules.
BLOCKED_PATHS = (DATA_DIR / "keys", DATA_DIR / ".hushed", DATA_DIR / "settings.json")


def command_block_reason(command: str) -> str | None:
    """Why *command* is refused, or None if it may run."""
    for blocked in BLOCKED_PATHS:
        if str(blocked) in command:
            logger.warning("blocked path in command %s: %s", blocked, command)
            return f"Command blocked: access to {blocked} is not permitted."
    return _blacklist_reason(command, "Command")


def path_block_reason(path: Path) -> str | None:
    """Why *path* may not be sent, or None if it may.  Symlinks are resolved first, so a
    link to a blocked file is refused too."""
    resolved = path.resolve()
    for blocked in BLOCKED_PATHS:
        if resolved == blocked.resolve() or resolved.is_relative_to(blocked.resolve()):
            logger.warning("blocked path refused %s (%s)", path, resolved)
            return f"Blocked: {path} is not permitted."
    return _blacklist_reason(str(resolved), "Path")


def _blacklist_reason(text: str, what: str) -> str | None:
    for pattern in load_settings().tools.bash.blacklist:
        if re.search(pattern, text):
            logger.warning("blacklist pattern %r matched: %s", pattern, text)
            return f"{what} blocked by blacklist pattern: {pattern}"
    return None
