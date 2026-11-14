"""Drives ``claude setup-token`` in a pseudo-terminal: it prints a sign-in link, reads
the code the sign-in page shows, and prints a one-year token.

The CLI only runs on a terminal, so it gets one.  It runs in a throwaway config
directory: the token is all that is kept, and no other Claude login is touched.
"""

import asyncio
import contextlib
import fcntl
import logging
import os
import re
import select
import shutil
import signal
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_URL_TIMEOUT = 30.0
_TOKEN_TIMEOUT = 60.0
# Wide enough that neither the link nor the token is wrapped.
_COLUMNS, _ROWS = 1000, 50
_ESCAPES = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?<>=]*[A-Za-z~]|\x1b[()][A-Z0-9]"
)
_URL = re.compile(r"https://\S+/oauth/authorize\?\S+")
_TOKEN = re.compile(r"sk-ant-oat\d+-[A-Za-z0-9_-]+")
# A refused code: the CLI says so, then waits for Enter to retry.
_TOKEN_OR_REFUSAL = re.compile(rf"(?P<token>{_TOKEN.pattern})|(?P<refused>OAuth error:[^\r\n]*)")
# The sign-in page shows "<code>#<state>".
_CODE = re.compile(r"^[A-Za-z0-9_-]+#[A-Za-z0-9_-]+$")


class SetupTokenError(Exception):
    """The login could not be started or finished."""


def looks_like_code(text: str) -> bool:
    """Whether *text* has the shape of the code the sign-in page shows."""
    return bool(_CODE.match(text.strip()))


class SetupToken:
    """One ``claude setup-token`` run: ``start()`` returns the sign-in link,
    ``finish(code)`` returns the token, ``close()`` ends it either way."""

    def __init__(self, cli: Path) -> None:
        self._cli = cli
        self._home = Path(tempfile.mkdtemp(prefix="claude-login-"))
        self._fd: int | None = None
        self._process: subprocess.Popen | None = None
        self._output = ""
        self._closed = False

    async def start(self) -> str:
        """Start the CLI and return the sign-in link it prints."""
        self._fd = await asyncio.to_thread(self._spawn)
        match = await asyncio.to_thread(self._read_until, _URL, _URL_TIMEOUT)
        if match is None:
            raise SetupTokenError(f"the Claude CLI printed no sign-in link: {self._tail()}")
        return match.group(0)

    async def finish(self, code: str) -> str:
        """Hand the CLI the sign-in code and return the token it prints."""
        if self._fd is None:
            raise SetupTokenError("the login was not started")
        self._output = ""
        os.write(self._fd, code.strip().encode())
        await asyncio.sleep(0.3)  # the CLI reads a paste and Enter apart
        os.write(self._fd, b"\r")
        match = await asyncio.to_thread(self._read_until, _TOKEN_OR_REFUSAL, _TOKEN_TIMEOUT)
        if match is None:
            # Not the CLI's output: it shows the code that was typed.
            raise SetupTokenError("Claude did not answer the code")
        if match["refused"]:
            raise SetupTokenError(f"Claude did not accept the code: {match['refused']}")
        return match["token"]

    async def close(self) -> None:
        """Stop the CLI and remove its config directory."""
        self._closed = True  # a read still running stops at its next pass
        if self._process is not None and self._process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGKILL)
            await asyncio.to_thread(self._process.wait)
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
        await asyncio.to_thread(shutil.rmtree, self._home, True)

    def _spawn(self) -> int:
        """Start the CLI on a new terminal; returns the terminal's controlling end."""
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", _ROWS, _COLUMNS, 0, 0))
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self._home),
            "CLAUDE_CONFIG_DIR": str(self._home),
            "TERM": "xterm-256color",
            "DISABLE_AUTOUPDATER": "1",
        }
        try:
            self._process = subprocess.Popen(
                [str(self._cli), "setup-token"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=env,
                cwd=self._home,
                start_new_session=True,
            )
        except OSError as e:
            os.close(master)
            raise SetupTokenError(f"could not start the Claude CLI: {e}") from e
        finally:
            os.close(slave)
        return master

    def _read_until(self, pattern: re.Pattern[str], timeout: float) -> re.Match[str] | None:
        """Read the terminal until *pattern* shows, the CLI exits, or *timeout* passes."""
        fd = self._fd
        assert fd is not None
        deadline = time.monotonic() + timeout
        raw = b""
        while time.monotonic() < deadline and not self._closed:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:  # the CLI exited and closed its end
                    chunk = b""
                if not chunk:
                    break
                raw += chunk
                self._output = _ESCAPES.sub("", raw.decode("utf-8", errors="replace"))
                # The token ends at the line end; wait for it so it is not cut short.
                match = pattern.search(self._output)
                if match and match.end() < len(self._output):
                    return match
            elif self._process is not None and self._process.poll() is not None:
                break
        match = pattern.search(self._output)
        return match if match and self._process and self._process.poll() is not None else None

    def _tail(self) -> str:
        """The CLI's last words, for an error before any code is typed (the token is
        stripped all the same)."""
        lines = [line.strip() for line in _TOKEN.sub("", self._output).splitlines()]
        return " / ".join(line for line in lines if line)[-300:] or "no output"
