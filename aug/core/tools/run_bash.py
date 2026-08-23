"""Bash execution tool with hushed secret injection and blacklist filtering."""

import logging
import os
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

# Tell common tools up front that nobody is at the keyboard, so they fail or take a
# default instead of prompting into a void.  Inherits the rest of the environment —
# hushed injects the user's secrets there.
_NONINTERACTIVE_ENV = {
    **os.environ,
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_NO_INPUT": "1",
}

# Substrings that mark output as "the command wanted a human".  ENOTTY is what a
# hidden-input read returns when stdin is not a terminal.
_INTERACTIVE_MARKERS = (
    "inappropriate ioctl for device",
    "not a tty",
    "no tty present",
    "terminal prompts disabled",
    "eof when reading a line",
)


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

    try:
        result = subprocess.run(
            ["hushed", "run", "--", "bash", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT,
            # capture_output only redirects stdout/stderr — without this, stdin is
            # inherited from the server process, which has no TTY.  Anything that
            # prompts then fails with ENOTTY ("inappropriate ioctl for device") and
            # retries in a loop instead of exiting.  DEVNULL turns a prompt into an
            # immediate EOF, so interactive commands fail fast and predictably.
            stdin=subprocess.DEVNULL,
            env=_NONINTERACTIVE_ENV,
        )
    except subprocess.TimeoutExpired:
        logger.warning("run_bash timed out after %ds cmd=%.120r", _TIMEOUT, command)
        return (
            f"Command did NOT complete: timed out after {_TIMEOUT}s. It may be waiting "
            f"for input — this shell is non-interactive, so prompts never get an answer. "
            f"Re-run with a non-interactive flag (e.g. -y, --yes, --no-input, --batch)."
        )
    except FileNotFoundError:
        logger.error("run_bash: hushed binary not found")
        return "Command did NOT run: the 'hushed' binary is not installed in this container."

    if result.returncode != 0:
        logger.warning(
            "run_bash exit_code=%d stderr=%.200r", result.returncode, result.stderr.strip()
        )
    else:
        logger.debug("run_bash exit_code=0")
    output = (result.stdout + result.stderr).strip()

    if result.returncode != 0 and _looks_interactive(output):
        return (
            f"Command did NOT complete: it tried to prompt for input, but this shell is "
            f"non-interactive so the prompt could never be answered. Supply the value on "
            f"the command line or use a non-interactive flag. Output:\n{output}"
        )
    if result.returncode != 0:
        return f"Command failed (exit {result.returncode}):\n{output or '(no output)'}"
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


def _looks_interactive(output: str) -> bool:
    """Return True if *output* shows the command failed by waiting on a prompt."""
    lowered = output.lower()
    return any(marker in lowered for marker in _INTERACTIVE_MARKERS)
