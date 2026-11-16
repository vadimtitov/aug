"""Thin client for the hushed secrets CLI."""

import asyncio
import logging
import re
import shlex
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

_LIST_TIMEOUT = 10
_READ_TIMEOUT = 15.0
_MAX_ERROR_LEN = 300
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# A tiny, argv-driven reader: `hushed run` injects the secret into this child's
# environment under NAME, and it writes the raw bytes straight to the fd number
# given on argv. Passing the fd as an int argument (not shell syntax) is what
# lets this work past descriptor 9 — see read_secret.
_READER_SCRIPT = (
    "import os,sys\n"
    "fd = int(sys.argv[1])\n"
    "os.write(fd, os.environb.get(sys.argv[2].encode(), b''))\n"
)
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


class HushedError(Exception):
    """A secret could not be read from hushed."""


def hushed_argv(command: str) -> list[str]:
    """The argv that runs *command* with the user's secrets injected and redacted."""
    return ["hushed", "run", "--", "bash", "-c", _WRAPPER, "bash", command]


def hushed_command(command: str) -> str:
    """*command* rewritten as one shell line that runs it through hushed_argv()."""
    return shlex.join(hushed_argv(command))


def unhushed(line: str) -> str | None:
    """The command inside *line* if it is a hushed_command(), else None."""
    prefix = hushed_argv("")[:-1]
    try:
        argv = shlex.split(line)
    except ValueError:
        return None
    if len(argv) == len(prefix) + 1 and argv[: len(prefix)] == prefix:
        return argv[-1]
    return None


async def list_secret_names() -> set[str]:
    """Names of every secret hushed currently holds, or empty on failure.

    Names only — hushed never reveals values via `list`.
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["hushed", "list"],
            capture_output=True,
            text=True,
            timeout=_LIST_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("hushed list failed: %r", exc)
        return set()
    if result.returncode != 0:
        logger.warning("hushed list exit_code=%d stderr=%.200r", result.returncode, result.stderr)
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def write_secret(name: str, value: str) -> None:
    """Store *value* in hushed under *name*, replacing any value it had.

    Raises:
        HushedError: the name is invalid or hushed failed.
    """
    if not _ENV_VAR_NAME_RE.match(name):
        raise HushedError(f"invalid secret name {name!r} — must be a valid env var name")
    try:
        result = subprocess.run(
            ["hushed", "add", name, value], capture_output=True, text=True, timeout=_LIST_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Not the exception itself: its text holds the command line, value included.
        raise HushedError(f"failed to store secret {name!r}: {type(exc).__name__}") from None
    if result.returncode != 0:
        raise HushedError(
            f"hushed failed storing {name!r}: {result.stderr.strip()[:_MAX_ERROR_LEN]}"
        )


def read_secret(name: str) -> str:
    """Fetch one secret's value from hushed via a private inherited pipe, since
    `hushed run` redacts secret values from stdout. The reader is a tiny Python
    child (fd passed as a plain argv int, not shell ``>&N`` redirection, which
    breaks past descriptor 9 on Debian's dash).

    Raises:
        HushedError: the name is invalid, hushed failed, or the secret is not set.
    """
    if not _ENV_VAR_NAME_RE.match(name):
        raise HushedError(f"invalid secret name {name!r} — must be a valid env var name")

    with tempfile.TemporaryFile() as tf:
        fd = tf.fileno()
        try:
            result = subprocess.run(
                ["hushed", "run", "--", sys.executable, "-c", _READER_SCRIPT, str(fd), name],
                pass_fds=(fd,),
                capture_output=True,
                text=True,
                timeout=_READ_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HushedError(f"failed to read secret {name!r}: {exc}") from exc
        if result.returncode != 0:
            raise HushedError(
                f"hushed failed resolving {name!r}: {result.stderr.strip()[:_MAX_ERROR_LEN]}"
            )
        tf.seek(0)
        value = tf.read().decode("utf-8", errors="replace")

    if not value:
        raise HushedError(f"secret {name!r} is not set. Set it with: hushed add {name} <value>")
    return value
