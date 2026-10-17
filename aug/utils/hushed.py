"""Thin client for the hushed secrets CLI."""

import asyncio
import logging
import subprocess

logger = logging.getLogger(__name__)

_LIST_TIMEOUT = 10


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
