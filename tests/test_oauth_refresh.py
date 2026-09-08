"""Tests for aug/core/oauth/refresh.py — on-demand token refresh.

This is the part of the design that forced tokens into Postgres rather than
hushed: rotating providers invalidate the old refresh token, so a lost update or
a double refresh disconnects the account permanently.

Behaviors under test:
  - concurrent callers for one account trigger exactly one token exchange
  - a rotated refresh token is what the next refresh uses
  - invalid_grant → account flagged for re-auth, row kept, no exception escapes
  - a reuse race is retried once rather than killing the account
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from aug.core.oauth.providers import ProviderConfig
from aug.core.oauth.refresh import TokenRefresher
from aug.core.oauth.store import decrypt, encrypt

_SPOTIFY = ProviderConfig(
    authorize_url="https://accounts.spotify.com/authorize",
    token_url="https://accounts.spotify.com/api/token",
    api_base="https://api.spotify.com",
    scopes=["user-read-private"],
)


class FakeConn:
    """A stand-in for the single oauth_tokens row, so refreshes actually take effect.

    A plain AsyncMock cannot express this: every read would return the stale row and
    a second refresh would look correct when it is exactly the bug under test.
    """

    def __init__(self, expires_in_seconds: int = 5):
        self.row = {
            "provider": "spotify",
            "account": "primary",
            "access_token_enc": encrypt("at-old", "spotify", "primary"),
            "refresh_token_enc": encrypt("rt-old", "spotify", "primary"),
            "token_type": "Bearer",
            "scopes": "",
            "expires_at": datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
            "needs_reauth": False,
            "last_error": None,
        }

    async def fetchrow(self, sql, *args):
        return self.row

    async def execute(self, sql, *args):
        if "needs_reauth = TRUE" in sql:
            self.row |= {"needs_reauth": True, "last_error": args[2]}
            return
        _provider, _account, access_enc, refresh_enc, token_type, scopes, expires_at = args
        self.row |= {
            "access_token_enc": access_enc,
            "refresh_token_enc": refresh_enc or self.row["refresh_token_enc"],
            "token_type": token_type,
            "scopes": scopes,
            "expires_at": expires_at,
        }


def _make_state(conn, handler):
    pool = MagicMock()
    cm = MagicMock()

    async def aenter(*_):
        return conn

    async def aexit(*_):
        return None

    cm.__aenter__, cm.__aexit__ = aenter, aexit
    pool.acquire.return_value = cm
    return SimpleNamespace(
        db_pool=pool,
        oauth_providers={"spotify": _SPOTIFY},
        oauth_transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def _client_credentials(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csecret")


async def test_concurrent_callers_refresh_exactly_once():
    """Five callers, one exchange. Two would invalidate each other on a rotating provider."""
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
        )

    conn = FakeConn()
    refresher = TokenRefresher(_make_state(conn, handler))

    results = await asyncio.gather(*(refresher.valid_token("spotify", "primary") for _ in range(5)))

    assert len(calls) == 1
    assert {token.access_token for token in results} == {"at-new"}
    assert decrypt(conn.row["refresh_token_enc"], "spotify", "primary") == "rt-new"


async def test_rotated_refresh_token_is_used_by_the_next_refresh():
    """The rotated token must be persisted, or the account dies at the next renewal."""
    sent: list[str] = []

    def handler(request):
        form = dict(httpx.QueryParams(request.content.decode()))
        sent.append(form["refresh_token"])
        return httpx.Response(
            200,
            json={"access_token": "at", "refresh_token": f"rt-{len(sent)}", "expires_in": 0},
        )

    conn = FakeConn()
    refresher = TokenRefresher(_make_state(conn, handler))

    await refresher.valid_token("spotify", "primary")
    await refresher.valid_token("spotify", "primary")

    assert sent == ["rt-old", "rt-1"]


async def test_invalid_grant_flags_the_account_and_keeps_the_row():
    """A revoked grant is terminal: flag it, keep the row so the failure is explainable."""

    def handler(request):
        return httpx.Response(400, json={"error": "invalid_grant"})

    conn = FakeConn()
    refresher = TokenRefresher(_make_state(conn, handler))

    token = await refresher.valid_token("spotify", "primary")

    assert token.needs_reauth is True
    assert "invalid_grant" in token.last_error
    assert conn.row["needs_reauth"] is True  # row retained, not deleted


async def test_refresh_token_reuse_is_retried_once():
    """A reuse race is transient — retry rather than declaring the account dead."""
    attempts: list[httpx.Request] = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(400, json={"error": "refresh_token_reused"})
        return httpx.Response(200, json={"access_token": "at-new", "expires_in": 3600})

    conn = FakeConn()
    refresher = TokenRefresher(_make_state(conn, handler))

    token = await refresher.valid_token("spotify", "primary")

    assert len(attempts) == 2
    assert token.access_token == "at-new"
    assert token.needs_reauth is False
