"""The ChatGPT login AUG's Codex agents run on, made by device code from chat.

The user opens a link, signs in to ChatGPT and enters a short code there; nothing is
pasted back into the chat.  Codex keeps the login (``auth.json`` in CODEX_HOME) and
refreshes it itself; the runtime refreshes it daily besides, so it never lapses for lack
of use.  A login that has expired or been revoked shows as the ``login`` error, whose
message names /codex_login.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from aug.core.harness.codex.runtime import CodexRuntime, codex_runtime
from aug.utils.codex_app_server import AppServerClosed, CodexRpcError

logger = logging.getLogger(__name__)

# How long a device code is good for at OpenAI; the wait gives up a little after.
LOGIN_TIMEOUT_SECONDS = 16 * 60


@dataclass(frozen=True)
class CodexAccount:
    """Who Codex is signed in as."""

    email: str
    plan: str


class LoginFailed(Exception):
    """The sign-in ended without a login: refused, expired, cancelled."""


class LoginAttempt:
    """One sign-in: open ``url``, enter ``code``; ``wait()`` for it to finish.
    ``cancel()`` ends it without a login."""

    def __init__(self, runtime: CodexRuntime, login_id: str, *, url: str, code: str) -> None:
        self.url = url
        self.code = code
        self._runtime = runtime
        self._login_id = login_id
        self._finished = runtime.login_finished(login_id)

    async def wait(self) -> CodexAccount:
        """Wait for the user to finish signing in.  Raises LoginFailed if they don't."""
        try:
            async with asyncio.timeout(LOGIN_TIMEOUT_SECONDS):
                error = await asyncio.shield(self._finished)
        except TimeoutError:
            await self.cancel()
            raise LoginFailed("the code expired") from None
        if error is not None:
            raise LoginFailed(error)
        try:
            account = await login_status(self._runtime)
        except Exception as e:  # signed in all the same: only who as is unknown
            logger.warning("codex login made, account unreadable: %r", e)
            return CodexAccount(email="", plan="")
        if account is None:
            raise LoginFailed("Codex reports no login after the sign-in")
        logger.info("codex login made plan=%s", account.plan)
        return account

    async def cancel(self) -> None:
        with contextlib.suppress(CodexRpcError, AppServerClosed, TimeoutError):
            await self._runtime.request("account/login/cancel", {"loginId": self._login_id})
        if not self._finished.done():
            self._finished.set_result("cancelled")


async def login_status(runtime: CodexRuntime = codex_runtime) -> CodexAccount | None:
    """Who Codex is signed in as; None when nobody is."""
    result = await runtime.request("account/read", {"refreshToken": False})
    return _account(result.get("account"))


async def start_login(runtime: CodexRuntime = codex_runtime) -> LoginAttempt:
    """Start a ChatGPT sign-in by device code.  Raises CodexRpcError, AppServerClosed,
    TimeoutError or OSError when Codex won't."""
    result = await runtime.request("account/login/start", {"type": "chatgptDeviceCode"})
    attempt = LoginAttempt(
        runtime,
        str(result["loginId"]),
        url=str(result["verificationUrl"]),
        code=str(result["userCode"]),
    )
    logger.info("codex sign-in started")
    return attempt


def _account(raw: Any) -> CodexAccount | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("type") == "chatgpt":
        return CodexAccount(email=str(raw.get("email") or ""), plan=str(raw.get("planType") or ""))
    return CodexAccount(email="", plan=str(raw.get("type") or ""))
