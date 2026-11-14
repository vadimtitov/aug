"""The Claude login AUG's Claude Code agents run on: a one-year ``claude setup-token``
token in hushed, renewed through a short sign-in started from chat.

A setup token never refreshes, so it never dies on a failed refresh, and any number of
Claude processes can share it.  When it was issued is kept in state.json: the harness
restarts its processes on a new login.  An expired or revoked login shows as the
``login`` error, whose message names /claude_login.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from aug.core.harness.claude_code.harness import BUNDLED_CLI, TOKEN_SECRET
from aug.utils.claude_setup_token import SetupToken
from aug.utils.hushed import write_secret
from aug.utils.state import load_state, update_state

logger = logging.getLogger(__name__)

LOGIN_LIFETIME = timedelta(days=365)


@dataclass(frozen=True)
class LoginStatus:
    """The login made through AUG, if any (a token set by hand has no known date)."""

    issued_at: datetime | None

    @property
    def expires_at(self) -> datetime | None:
        return self.issued_at + LOGIN_LIFETIME if self.issued_at else None


def login_status() -> LoginStatus:
    issued = load_state().claude_login.issued_at
    return LoginStatus(issued_at=datetime.fromisoformat(issued) if issued else None)


class LoginAttempt:
    """One sign-in: ``url`` to open, then ``complete(code)`` with the code the sign-in
    page shows.  ``cancel()`` ends it without a login."""

    def __init__(self, setup: SetupToken, url: str) -> None:
        self._setup = setup
        self.url = url

    async def complete(self, code: str) -> LoginStatus:
        """Exchange *code* for a token and make it the login.  The token never leaves
        hushed.  Raises SetupTokenError (code refused) or HushedError (not stored)."""
        try:
            token = await self._setup.finish(code)
        finally:
            await self._setup.close()
        await asyncio.to_thread(write_secret, TOKEN_SECRET, token)
        issued = datetime.now(UTC)
        async with update_state() as state:
            state.claude_login.issued_at = issued.isoformat()
        logger.info("claude login renewed, expires %s", (issued + LOGIN_LIFETIME).date())
        return LoginStatus(issued_at=issued)

    async def cancel(self) -> None:
        await self._setup.close()


async def start_login() -> LoginAttempt:
    """Start a sign-in.  Raises SetupTokenError when the Claude CLI won't start one."""
    setup = SetupToken(BUNDLED_CLI)
    try:
        url = await setup.start()
    except BaseException:
        await setup.close()
        raise
    return LoginAttempt(setup, url)
