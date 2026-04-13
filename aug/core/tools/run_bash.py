"""Bash execution tool with hushed secret injection and blacklist filtering."""

import logging
import re
import subprocess

from langchain_core.tools import tool

from aug.utils.data import DATA_DIR
from aug.utils.file_settings import load_settings

# SSH private keys must never be readable by the agent, regardless of
# user-configured blacklist entries.
_KEYS_DIR = str(DATA_DIR / "keys")

logger = logging.getLogger(__name__)

_TIMEOUT = 60


@tool
def run_bash(command: str) -> str:
    """Execute a shell command inside the container.

    SECRETS: The user may have stored secrets (API keys, passwords, tokens, urls,etc.)
    using a tool called hushed. You have no visibility into what secrets exist
    until you ask. To discover available secrets, run: hushed list
    This returns a list of names like: OPENAI_API_KEY, GITHUB_TOKEN, etc.
    Each secret is injected as an environment variable under that exact name,
    so you can reference it in commands as $SECRET_NAME.
    Secret values are never visible — they are automatically redacted from output.

    Always run `hushed list` first if a command might need credentials.

    Args:
        command: Shell command to run.
    """
    if error := _check_blacklist(command):
        return error

    logger.info("run_bash cmd=%.120r", command)

    result = subprocess.run(
        ["hushed", "run", "--", "bash", "-c", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        logger.warning(
            "run_bash exit_code=%d stderr=%.200r", result.returncode, result.stderr.strip()
        )
    else:
        logger.debug("run_bash exit_code=0")
    output = (result.stdout + result.stderr).strip()
    return output or "(no output)"


def _check_blacklist(command: str) -> str | None:
    """Return an error string if the command matches a blacklist pattern, else None."""
    if _KEYS_DIR in command:
        logger.warning("run_bash blocked keys dir access: %s", command)
        return f"Command blocked: access to {_KEYS_DIR} is not permitted."
    patterns = load_settings().tools.bash.blacklist
    for pattern in patterns:
        if re.search(pattern, command):
            logger.warning("run_bash blocked by blacklist pattern %r: %s", pattern, command)
            return f"Command blocked by blacklist pattern: {pattern}"
    return None
