"""Bash execution tool with hushed secret injection and blacklist filtering."""

import asyncio
import logging
import os
import re
import signal

from langchain_core.tools import tool

from aug.utils.data import DATA_DIR
from aug.utils.file_settings import load_settings

# SSH private keys must never be readable by the agent, regardless of
# user-configured blacklist entries.
_KEYS_DIR = str(DATA_DIR / "keys")

logger = logging.getLogger(__name__)

# Runs the command ($1) with its output in temp files, then prints them.  hushed relays
# (and redacts) output only once every process holding it has let go, so a background
# job started by the command would otherwise keep the call open until it exits.  With
# the files, it inherits those instead, and the call returns when the command does.
#
# The files hold output before hushed redacts it, so they are deleted as soon as they
# are open and read back through the open descriptors: a /stop (SIGKILL) leaves nothing
# on disk.  They live in /dev/shm (RAM) where there is one, so a full disk can't swallow
# the output of the very command that diagnoses it.  If no file can be made, the command
# runs on the pipe directly, as before.
_WRAPPER = """\
d=/dev/shm; [ -d "$d" ] && [ -w "$d" ] || d=${TMPDIR:-/tmp}
o=$(mktemp "$d/aug-bash.XXXXXX") && e=$(mktemp "$d/aug-bash.XXXXXX") || exec bash -c "$1"
exec 3>"$o" 4<"$o" 5>"$e" 6<"$e"; rm -f "$o" "$e"
bash -c "$1" >&3 2>&5 3>&- 4<&- 5>&- 6<&-; s=$?
exec 3>&- 5>&-; cat <&4; cat <&6 >&2; exit $s
"""


@tool
async def run_bash(command: str) -> str:
    """Execute a shell command inside the container.

    SECRETS: The user may have stored secrets (API keys, passwords, tokens, urls,etc.)
    using a tool called hushed. You have no visibility into what secrets exist
    until you ask. To discover available secrets, run: hushed list
    This returns a list of names like: OPENAI_API_KEY, GITHUB_TOKEN, etc.
    Each secret is injected as an environment variable under that exact name,
    so you can reference it in commands as $SECRET_NAME.
    Secret values are never visible — they are automatically redacted from output.

    Always run `hushed list` first if a command might need credentials.

    BACKGROUND JOBS: the call returns as soon as the command itself exits. To leave
    something running after that (a server, a long job), start it with
    `setsid nohup CMD > /tmp/CMD.log 2>&1 &` so it neither holds the output open nor
    dies if the user stops this call.

    Args:
        command: Shell command to run.
    """
    if error := _check_blacklist(command):
        return error

    logger.info("run_bash cmd=%.120r", command)

    # Its own process group, so /stop (which cancels this coroutine) kills the command
    # and everything it started, not just the hushed wrapper.  No timeout: long
    # commands are fine, and /stop is how they end early.
    proc = await asyncio.create_subprocess_exec(
        "hushed",
        "run",
        "--",
        "bash",
        "-c",
        _WRAPPER,
        "bash",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        _kill_group(proc)
        await proc.wait()  # reap it; SIGKILL makes this immediate
        raise
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        logger.warning("run_bash exit_code=%d stderr=%.200r", proc.returncode, err.strip())
    else:
        logger.debug("run_bash exit_code=0")
    output = (out + err).strip()
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


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the command's whole process group, background jobs included.  Processes
    that left it with setsid (or docker run -d) survive."""
    if proc.returncode is not None:
        return  # already reaped: its pid may belong to someone else by now
    logger.info("run_bash killed pid=%d", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
