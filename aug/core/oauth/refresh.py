"""On-demand token refresh.

Refresh happens at the point of use, never on a background loop: a loop renews
credentials nobody is using, multiplying exposure to rotation failures for no
benefit, and still cannot save you from a token the provider killed early.
"""

import asyncio
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx

from aug.core.oauth.providers import ProviderConfig
from aug.core.oauth.store import StoredToken, load_token, mark_needs_reauth, save_token
from aug.utils.oauth import OAuthClient

logger = logging.getLogger(__name__)

# Refresh this far ahead of expiry so a call in flight cannot land on a dead token.
_SKEW = timedelta(seconds=60)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class TokenRefresher:
    """Hands out live tokens, renewing them when they are about to expire.

    Owns one lock per (provider, account): concurrent callers for the same account
    must not each start a refresh, because a provider that rotates would invalidate
    the winner's token and leave the account disconnected.
    """

    def __init__(self, state) -> None:
        self._state = state
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def valid_token(self, provider: str, account: str) -> StoredToken | None:
        """Return a usable token, refreshing first if it is close to expiry."""
        async with self._state.db_pool.acquire() as conn:
            token = await load_token(conn, provider, account)

        if token is None or token.needs_reauth or not _expiring(token):
            return token

        async with self._lock_for(provider, account):
            async with self._state.db_pool.acquire() as conn:
                # Re-read inside the lock: a waiter that arrives after someone else
                # refreshed simply gets the fresh token instead of rotating again.
                token = await load_token(conn, provider, account)
                if token is None or not _expiring(token) or token.needs_reauth:
                    return token
                return await self._refresh(conn, token)

    def _lock_for(self, provider: str, account: str) -> asyncio.Lock:
        return self._locks.setdefault((provider, account), asyncio.Lock())

    async def _mark_dead(self, conn, token: StoredToken, error: str) -> StoredToken:
        """Record a dead grant. The row stays so the gateway can explain the failure."""
        logger.error(
            "oauth refresh failed provider=%s account=%s error=%s",
            token.provider,
            token.account,
            error,
        )
        await mark_needs_reauth(conn, token.provider, token.account, error)
        return replace(token, needs_reauth=True, last_error=error)

    async def _refresh(self, conn, token: StoredToken) -> StoredToken | None:
        """Perform the refresh and persist the result, including a rotated token."""
        config: ProviderConfig = self._state.oauth_providers.get(token.provider)
        client_id = os.environ.get(
            config.client_id_env or f"{token.provider.upper()}_CLIENT_ID", ""
        )
        secret = os.environ.get(
            config.client_secret_env or f"{token.provider.upper()}_CLIENT_SECRET", ""
        )

        async with httpx.AsyncClient(
            transport=self._state.oauth_transport, timeout=_TIMEOUT, follow_redirects=False
        ) as http:
            client = OAuthClient(config, client_id, secret, http)
            try:
                new = await client.refresh(token.refresh_token)
            except httpx.HTTPStatusError as exc:
                error = _error_code(exc)
                if _is_reuse_race(error):
                    # Another refresh landed first; the provider has a newer token
                    # than we do. One retry, then treat it as a real failure.
                    logger.warning(
                        "oauth refresh reuse race provider=%s — retrying", token.provider
                    )
                    new = await client.refresh(token.refresh_token)
                else:
                    return await self._mark_dead(conn, token, error)

        await save_token(conn, token.provider, token.account, new)
        logger.info("oauth refreshed provider=%s account=%s", token.provider, token.account)
        return StoredToken(
            provider=token.provider,
            account=token.account,
            access_token=new.access_token,
            # A provider that does not rotate omits the refresh token; keep the old one.
            refresh_token=new.refresh_token or token.refresh_token,
            token_type=new.token_type,
            expires_at=new.expires_at,
            needs_reauth=False,
            last_error=None,
        )


def _error_code(exc: httpx.HTTPStatusError) -> str:
    """The provider's OAuth error code, falling back to the raw body."""
    try:
        payload = exc.response.json()
    except ValueError:
        return exc.response.text[:200]
    return payload.get("error") or payload.get("error_description") or exc.response.text[:200]


def _is_reuse_race(error: str) -> bool:
    """Distinguish a lost rotation race from a genuinely revoked grant."""
    lowered = error.lower()
    return "reused" in lowered or "already been used" in lowered


def _expiring(token: StoredToken) -> bool:
    """True if the token is close enough to expiry that it must be renewed now."""
    if token.expires_at is None or token.refresh_token is None:
        return False
    return token.expires_at - _SKEW <= datetime.now(UTC)
