"""Thin client for the hushed secrets CLI."""

import asyncio
import logging
import re
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


class HushedError(Exception):
    """A secret could not be read from hushed."""


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
