"""Tests for the public OAuth flow — aug/api/routers/oauth.py.

Behaviors under test:
  - valid state → code exchanged at the token endpoint, token stored encrypted
  - unknown or expired state → 400, and no outbound request is ever made
  - callback path provider ≠ state row provider → 400 (RFC 9700 mix-up defence)
  - iss present and wrong → 400; iss absent → accepted (RFC 9207)
  - provider dropped from the registry mid-flow → 400, not a 500
  - /start without a valid single-use token → 400, no authorization redirect
  - /start with a valid token → 302 carrying PKCE S256 and the exact redirect URI
  - callback pages never leak the code via Referer or caches
  - both public endpoints are rate limited per client IP, ahead of any DB work
  - the user declining consent gets a plain page, not a validation error
  - /start's path segment must agree with the token it was minted for
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app
from aug.core.oauth.providers import ProviderConfig, ProviderRegistry

_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)

_SPOTIFY = ProviderConfig(
    authorize_url="https://accounts.spotify.com/authorize",
    token_url="https://accounts.spotify.com/api/token",
    api_base="https://api.spotify.com",
    scopes=["user-read-private"],
)


def _state_row(**overrides):
    row = {
        "state": "state-abc",
        "provider": "spotify",
        "account": "primary",
        "code_verifier": "verifier-xyz",
        "redirect_uri": "https://aug.example.com/oauth/spotify/callback",
        "issuer": None,
        "expires_at": _NOW + timedelta(minutes=5),
    }
    row.update(overrides)
    return row


def _make_pool(conn):
    """Mock asyncpg pool whose acquire() context manager yields conn."""
    pool = MagicMock()
    pool.close = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool


def _async_ctx(value):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=value)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.fixture()
def conn():
    return AsyncMock()


@pytest.fixture()
def upstream():
    """Records requests made to the provider and replies with a token response."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "at-secret-value",
                "refresh_token": "rt-secret-value",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "user-read-private",
            },
        )

    transport = httpx.MockTransport(handler)
    transport.requests = requests  # type: ignore[attr-defined]
    return transport


@pytest.fixture()
def providers(tmp_path):
    """A registry backed by a real file, so tests exercise the actual load path."""
    registry = ProviderRegistry(tmp_path / "oauth_providers.json")
    registry.save("spotify", _SPOTIFY.model_dump(exclude_none=True))
    return registry


@pytest.fixture()
def client(conn, upstream, providers, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csecret")
    pool = _make_pool(conn)
    with (
        patch("aug.app.create_pool", new=AsyncMock(return_value=pool)),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(MagicMock())),
        patch("aug.app.init_memory_files"),
        patch("aug.app.start_consolidation_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.start_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch("aug.app.stop_scheduler", new=AsyncMock()),
        patch("aug.app.set_push_app"),
        patch("aug.app.set_pool"),
        patch("aug.app.serve_gateway", new=AsyncMock()),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.oauth_providers = providers
            app.state.oauth_transport = upstream
            yield c


def test_callback_exchanges_code_and_stores_encrypted_token(client, conn, upstream):
    """A valid state claims the row, exchanges the code, and persists the token."""
    conn.fetchrow = AsyncMock(return_value=_state_row())

    response = client.get(
        "/oauth/spotify/callback",
        params={"code": "auth-code-1", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.status_code == 200

    # Exactly one call, to the configured token endpoint.
    assert len(upstream.requests) == 1
    token_request = upstream.requests[0]
    assert str(token_request.url) == "https://accounts.spotify.com/api/token"
    form = dict(httpx.QueryParams(token_request.content.decode()))
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "auth-code-1"
    assert form["code_verifier"] == "verifier-xyz"
    assert form["redirect_uri"] == "https://aug.example.com/oauth/spotify/callback"

    # The token reached the database, and not in the clear.
    assert conn.execute.await_count == 1
    stored = conn.execute.await_args.args
    assert not any("at-secret-value" in str(arg) for arg in stored)
    assert any(isinstance(arg, bytes) for arg in stored)


def test_unknown_state_is_rejected_without_contacting_the_provider(client, conn, upstream):
    """An unrecognised state costs one indexed DELETE and nothing else.

    This is the first line of defence, ahead of rate limiting: an attacker cannot
    use the callback to make AUG generate outbound traffic.
    """
    conn.fetchrow = AsyncMock(return_value=None)

    response = client.get(
        "/oauth/spotify/callback",
        params={"code": "auth-code-1", "state": "not-a-real-state"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert upstream.requests == []
    assert conn.execute.await_count == 0


def test_provider_mismatch_between_path_and_state_is_rejected(client, conn, upstream):
    """A state minted for Spotify must not be usable at Strava's callback.

    RFC 9700 mix-up defence: without it, an authorization server the client also
    talks to can have a code redeemed at the wrong token endpoint.
    """
    conn.fetchrow = AsyncMock(return_value=_state_row(provider="spotify"))

    response = client.get(
        "/oauth/strava/callback",
        params={"code": "auth-code-1", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert upstream.requests == []
    assert conn.execute.await_count == 0


def test_iss_mismatch_is_rejected(client, conn, upstream):
    """RFC 9207: when the provider echoes iss, it must match the expected issuer."""
    conn.fetchrow = AsyncMock(return_value=_state_row(issuer="https://accounts.spotify.com"))

    response = client.get(
        "/oauth/spotify/callback",
        params={"code": "auth-code-1", "state": "state-abc", "iss": "https://evil.example"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert upstream.requests == []
    assert conn.execute.await_count == 0


def test_absent_iss_is_accepted(client, conn, upstream):
    """Most consumer providers omit iss entirely — its absence must not break the flow."""
    conn.fetchrow = AsyncMock(return_value=_state_row(issuer="https://accounts.spotify.com"))

    response = client.get(
        "/oauth/spotify/callback",
        params={"code": "auth-code-1", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert len(upstream.requests) == 1


def test_provider_missing_from_registry_fails_cleanly(client, conn, upstream):
    """A provider removed from config after the state was minted must not 500."""
    conn.fetchrow = AsyncMock(return_value=_state_row(provider="strava"))

    response = client.get(
        "/oauth/strava/callback",
        params={"code": "auth-code-1", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert upstream.requests == []


# ── /start ────────────────────────────────────────────────────────────────────


def test_start_without_a_valid_token_does_not_redirect(client, conn):
    """The start endpoint is browser-reachable, so it is gated by a minted token.

    Without this, an attacker can walk the user through a consent screen of their
    choosing and graft their own account onto AUG.
    """
    conn.fetchrow = AsyncMock(return_value=None)

    response = client.get("/oauth/spotify/start", params={"t": "bogus"}, follow_redirects=False)

    assert response.status_code == 400
    assert conn.execute.await_count == 0


def test_start_redirects_with_pkce_and_exact_redirect_uri(client, conn):
    """A valid start token mints state + PKCE and sends the browser to the provider."""
    conn.fetchrow = AsyncMock(return_value={"provider": "spotify", "account": "primary"})

    response = client.get(
        "/oauth/spotify/start", params={"t": "good-token"}, follow_redirects=False
    )

    assert response.status_code == 302
    location = httpx.URL(response.headers["location"])
    assert str(location.copy_with(query=None)) == "https://accounts.spotify.com/authorize"
    query = dict(location.params)
    assert query["response_type"] == "code"
    assert query["client_id"] == "cid"
    assert query["code_challenge_method"] == "S256"
    assert query["redirect_uri"] == "https://aug.test/oauth/spotify/callback"
    assert query["scope"] == "user-read-private"
    assert len(query["state"]) >= 32

    # The verifier is stored server-side, never sent to the browser.
    assert conn.execute.await_count == 1
    stored = conn.execute.await_args.args
    assert query["state"] in stored
    assert query["code_challenge"] not in stored


# ── code leakage ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state_row", [_state_row(), None], ids=["success", "failure"])
def test_callback_page_cannot_leak_the_code(client, conn, state_row):
    """The authorization code sits in the URL, so the page must not carry it outward.

    Any external asset would send the full URL as a Referer; any cache would retain
    it. Both success and failure pages are covered — a failure page is rendered for
    a request whose URL still contains a live code.
    """
    conn.fetchrow = AsyncMock(return_value=state_row)

    response = client.get(
        "/oauth/spotify/callback",
        params={"code": "auth-code-1", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-security-policy"] == "default-src 'none'"
    assert "//" not in response.text  # no external assets of any kind


# ── rate limiting ─────────────────────────────────────────────────────────────


def test_callback_is_rate_limited_before_touching_the_database(client, conn):
    """These endpoints are unauthenticated and internet-facing; flooding must be cheap
    to refuse. The limit is checked before the state lookup, so a flood costs no
    database work at all."""
    conn.fetchrow = AsyncMock(return_value=None)

    codes = [
        client.get(
            "/oauth/spotify/callback",
            params={"code": "c", "state": "s"},
            follow_redirects=False,
        ).status_code
        for _ in range(40)
    ]

    assert 429 in codes
    assert conn.fetchrow.await_count < 40


def test_client_ip_comes_from_the_rightmost_forwarded_for_entry(client, conn):
    """With one trusted proxy hop, everything left of the last entry is caller-forged.

    Taking the leftmost would let an attacker mint a fresh allowance per request by
    prepending a random address.
    """
    conn.fetchrow = AsyncMock(return_value=None)

    def call(forwarded_for: str) -> int:
        return client.get(
            "/oauth/spotify/callback",
            params={"code": "c", "state": "s"},
            headers={"X-Forwarded-For": forwarded_for},
            follow_redirects=False,
        ).status_code

    # Same real client, forged left-hand entries: still one allowance.
    codes = [call(f"{i}.{i}.{i}.{i}, 9.9.9.9") for i in range(1, 40)]
    assert 429 in codes

    # A genuinely different client, seen by the same proxy, is unaffected.
    assert call("1.1.1.1, 8.8.8.8") != 429


# ── the user declines ─────────────────────────────────────────────────────────


def test_declined_consent_renders_a_plain_page_and_releases_the_state(client, conn, upstream):
    """Cancelling is a normal outcome, not a crash.

    The provider sends back an error and no code, so a callback that requires `code`
    fails validation and shows the user a raw JSON blob after they deliberately
    declined. The pending state is consumed too, rather than left to expire.
    """
    conn.fetchrow = AsyncMock(return_value=_state_row())

    response = client.get(
        "/oauth/spotify/callback",
        params={"error": "access_denied", "state": "state-abc"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "not connected" in response.text.lower()
    assert conn.fetchrow.await_count == 1  # state claimed
    assert upstream.requests == []
    assert conn.execute.await_count == 0  # nothing stored


def test_callback_without_code_or_error_is_refused(client, conn, upstream):
    """A missing code must not reach the token exchange as None."""
    conn.fetchrow = AsyncMock(return_value=_state_row())

    response = client.get(
        "/oauth/spotify/callback", params={"state": "state-abc"}, follow_redirects=False
    )

    assert response.status_code == 400
    assert upstream.requests == []


def test_start_path_segment_must_match_the_minted_token(client, conn):
    """The link the user reads must not name a different provider than it connects.

    The token is authoritative, so this changes no behaviour — it stops the URL
    from lying about which account is about to be connected.
    """
    conn.fetchrow = AsyncMock(return_value={"provider": "spotify", "account": "primary"})

    response = client.get("/oauth/strava/start", params={"t": "good-token"}, follow_redirects=False)

    assert response.status_code == 400
    assert conn.execute.await_count == 0
