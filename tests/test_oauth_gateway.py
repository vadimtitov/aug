"""Tests for aug/api/internal/gateway.py — the loopback token gateway.

The gateway is how the agent uses a credential without being able to read it: it
calls ``localhost:8799/{provider}/{path}`` and the gateway attaches the token.

Behaviors under test:
  - a proxied call reaches the provider's pinned api_base with Authorization attached
  - provider configured but never connected → actionable error, no upstream call
  - a dead grant → 503 naming the failure, never a false success
  - the host is pinned: traversal and absolute paths cannot redirect the token
  - upstream redirects are returned, not followed
  - a client-supplied Authorization header cannot shadow the real one
  - GET / answers what is connected, including when nothing is
  - an expiring token is refreshed before the call, and the new one is persisted
  - the agent can mint a start link for the user to tap
  - disconnect revokes where possible, and never implies a revocation that did not happen
  - provider config is written through a validating endpoint, never by hand
  - a token that cannot be decrypted names the cause instead of a 500
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from aug.api.internal.gateway import create_gateway_app
from aug.core.oauth.providers import ProviderConfig, ProviderRegistry
from aug.core.oauth.store import encrypt

_STRAVA = ProviderConfig(
    authorize_url="https://www.strava.com/oauth/authorize",
    token_url="https://www.strava.com/oauth/token",
    api_base="https://www.strava.com/api",
    revoke_url="https://www.strava.com/oauth/deauthorize",
    scopes=["read"],
)

_SPOTIFY = ProviderConfig(
    authorize_url="https://accounts.spotify.com/authorize",
    token_url="https://accounts.spotify.com/api/token",
    api_base="https://api.spotify.com",
    scopes=["user-read-private"],
)


def _token_row(provider="spotify", account="primary", **overrides):
    row = {
        "provider": provider,
        "account": account,
        "access_token_enc": encrypt("at-live", provider, account),
        "refresh_token_enc": encrypt("rt-live", provider, account),
        "token_type": "Bearer",
        "scopes": "user-read-private",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
        "needs_reauth": False,
        "last_error": None,
    }
    row.update(overrides)
    return row


def _make_pool(conn):
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool


@pytest.fixture()
def conn():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=_token_row())
    conn.fetchval = AsyncMock(return_value=0)  # no live connections unless a test says so
    return conn


@pytest.fixture()
def upstream():
    """Records what the gateway forwarded, and replies 200."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "vadim"})

    transport = httpx.MockTransport(handler)
    transport.requests = requests  # type: ignore[attr-defined]
    return transport


@pytest.fixture()
def providers(tmp_path):
    """A registry backed by a real file, so tests exercise the actual load path."""
    registry = ProviderRegistry(tmp_path / "oauth_providers.json")
    registry.save("spotify", _SPOTIFY.model_dump(exclude_none=True))
    registry.save("strava", _STRAVA.model_dump(exclude_none=True))
    return registry


@pytest.fixture()
def client(conn, upstream, providers):
    state = SimpleNamespace(
        db_pool=_make_pool(conn),
        oauth_providers=providers,
        oauth_transport=upstream,
    )
    with TestClient(create_gateway_app(state), raise_server_exceptions=True) as c:
        yield c


def test_proxied_call_reaches_pinned_host_with_token_attached(client, upstream):
    """The agent calls the gateway; the provider sees an authenticated request."""
    response = client.get("/spotify/v1/me")

    assert response.status_code == 200
    assert response.json() == {"id": "vadim"}

    assert len(upstream.requests) == 1
    forwarded = upstream.requests[0]
    assert str(forwarded.url) == "https://api.spotify.com/v1/me"
    assert forwarded.headers["authorization"] == "Bearer at-live"


def test_provider_not_connected_says_so_and_does_not_call_upstream(client, conn, upstream):
    """A configured-but-unconnected provider must not read as a provider outage.

    Per the tool standard, the string the agent reads has to name the fix — an
    ambiguous failure here gets reported to the user as "Spotify is down".
    """
    conn.fetchrow = AsyncMock(return_value=None)

    response = client.get("/spotify/v1/me")

    assert response.status_code == 401
    assert "not connected" in response.text
    assert "spotify" in response.text
    assert upstream.requests == []


def test_dead_grant_returns_an_actionable_error(client, conn, upstream):
    """A revoked grant must fail loudly, with the reconnect instruction attached."""
    conn.fetchrow = AsyncMock(
        return_value=_token_row(needs_reauth=True, last_error="invalid_grant")
    )

    response = client.get("/spotify/v1/me")

    assert response.status_code == 503
    assert "invalid_grant" in response.text
    assert "re-authoriz" in response.text.lower()
    assert upstream.requests == []


@pytest.mark.parametrize(
    "path",
    [
        "/spotify/../../evil",
        "/spotify/..%2f..%2fevil",
        "/spotify/https://evil.example/steal",
        "/spotify//evil.example/steal",
    ],
)
def test_host_is_pinned_regardless_of_path(client, upstream, path):
    """The token is bound to one host. No path may send it anywhere else.

    Without this the gateway is just a bearer token with extra steps: an injected
    agent asks it to call an attacker's host and the credential walks out.
    """
    response = client.get(path)

    for forwarded in upstream.requests:
        assert forwarded.url.host == "api.spotify.com", f"{path} escaped to {forwarded.url}"
    assert response.status_code < 500


def test_upstream_redirect_is_returned_not_followed(client, conn, upstream):
    """Following a 3xx would carry the Authorization header off the pinned host."""

    def handler(request: httpx.Request) -> httpx.Response:
        upstream.requests.append(request)
        return httpx.Response(302, headers={"location": "https://evil.example/steal"})

    upstream.handler = handler

    response = client.get("/spotify/v1/me", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "https://evil.example/steal"
    assert len(upstream.requests) == 1


def test_client_supplied_authorization_is_stripped(client, upstream):
    """The agent must not be able to override the credential the gateway attaches."""
    response = client.get(
        "/spotify/v1/me",
        headers={"Authorization": "Bearer agent-chosen", "Cookie": "session=abc"},
    )

    assert response.status_code == 200
    forwarded = upstream.requests[0]
    assert forwarded.headers["authorization"] == "Bearer at-live"
    assert "cookie" not in forwarded.headers


def test_status_lists_connected_and_unconnected_providers(client, conn):
    """The gateway always listens, so this is the one place status is answered.

    It must distinguish "nothing connected" from "gateway is down" — the agent
    cannot tell those apart from a refused connection.
    """
    conn.fetch = AsyncMock(
        return_value=[{"provider": "spotify", "account": "primary", "needs_reauth": False}]
    )

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["connected"] == [{"provider": "spotify", "account": "primary", "healthy": True}]
    assert body["configured"] == ["spotify", "strava"]


def test_status_with_nothing_connected(client, conn):
    conn.fetch = AsyncMock(return_value=[])

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["connected"] == []


# ── refresh ───────────────────────────────────────────────────────────────────


def test_expiring_token_is_refreshed_before_the_call(client, conn, upstream):
    """A token about to expire is renewed in-line, so the agent never sees a 401.

    The rotated refresh token must also be persisted: providers like Strava
    invalidate the old one, so failing to store it disconnects the account for good.
    """
    conn.fetchrow = AsyncMock(
        return_value=_token_row(expires_at=datetime.now(UTC) + timedelta(seconds=5))
    )

    def handler(request: httpx.Request) -> httpx.Response:
        upstream.requests.append(request)
        if request.url.host == "accounts.spotify.com":
            return httpx.Response(
                200,
                json={
                    "access_token": "at-fresh",
                    "refresh_token": "rt-rotated",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(200, json={"id": "vadim"})

    upstream.handler = handler

    response = client.get("/spotify/v1/me")

    assert response.status_code == 200

    token_call, api_call = upstream.requests
    assert str(token_call.url) == "https://accounts.spotify.com/api/token"
    assert dict(httpx.QueryParams(token_call.content.decode()))["grant_type"] == "refresh_token"
    assert api_call.headers["authorization"] == "Bearer at-fresh"

    # The rotated refresh token was written back, encrypted.
    assert conn.execute.await_count == 1
    stored = conn.execute.await_args.args
    assert not any("rt-rotated" in str(arg) for arg in stored)


# ── minting start links ───────────────────────────────────────────────────────


def test_link_mints_a_single_use_start_url(client, conn):
    """The agent mints the link it sends the user; the token is stored server-side.

    Reserved ``/_link/`` prefix rather than ``/{provider}/link`` so a provider whose
    own API has a ``/link`` endpoint cannot be silently shadowed.
    """
    response = client.post("/_link/spotify")

    assert response.status_code == 200
    url = httpx.URL(response.json()["url"])
    assert str(url.copy_with(query=None)) == "https://aug.test/oauth/spotify/start"
    token = dict(url.params)["t"]
    assert len(token) >= 32

    assert conn.execute.await_count == 1
    stored = conn.execute.await_args.args
    assert token in stored
    assert "spotify" in stored


def test_link_for_unknown_provider_is_refused(client, conn):
    response = client.post("/_link/nope")

    assert response.status_code == 404
    assert "spotify" in response.text  # names what does exist
    assert conn.execute.await_count == 0


# ── disconnect ────────────────────────────────────────────────────────────────


def test_disconnect_revokes_then_deletes(client, conn, upstream):
    """A provider with a revoke endpoint gets a real disconnect, not just a local delete."""
    conn.fetchrow = AsyncMock(return_value=_token_row(provider="strava"))

    response = client.delete("/strava")

    assert response.status_code == 200
    assert str(upstream.requests[0].url) == "https://www.strava.com/oauth/deauthorize"
    assert conn.execute.await_count == 1
    assert "DELETE FROM oauth_tokens" in conn.execute.await_args.args[0]


def test_disconnect_without_revoke_support_says_access_was_not_revoked(client, conn, upstream):
    """Deleting the local row is not revocation. Saying otherwise would be a lie.

    Spotify has no revoke endpoint at all — the grant stays live until the user
    removes it themselves, and the agent must tell them that.
    """
    conn.fetchrow = AsyncMock(return_value=_token_row())

    response = client.delete("/spotify")

    assert response.status_code == 200
    assert "NOT revoked" in response.text
    assert upstream.requests == []
    assert conn.execute.await_count == 1


def test_disconnect_deletes_locally_even_when_revocation_fails(client, conn, upstream):
    """Otherwise a provider outage leaves a dead row that can never be removed."""
    conn.fetchrow = AsyncMock(return_value=_token_row(provider="strava"))
    upstream.handler = lambda request: httpx.Response(500)

    response = client.delete("/strava")

    assert response.status_code == 200
    assert "NOT revoked" in response.text
    assert conn.execute.await_count == 1


def test_disconnect_when_nothing_is_connected(client, conn, upstream):
    conn.fetchrow = AsyncMock(return_value=None)

    response = client.delete("/spotify")

    assert response.status_code == 404
    assert conn.execute.await_count == 0


# ── provider configuration ────────────────────────────────────────────────────

_TODOIST = {
    "authorize_url": "https://todoist.com/oauth/authorize",
    "token_url": "https://todoist.com/oauth/access_token",
    "api_base": "https://api.todoist.com",
    "scopes": ["data:read"],
}


def test_saving_a_provider_makes_it_immediately_usable(client, providers):
    """No restart between writing the config and minting a link for it."""
    response = client.post("/_provider/todoist", json=_TODOIST)

    assert response.status_code == 200
    assert providers.get("todoist") is not None
    assert client.post("/_link/todoist").status_code == 200


def test_invalid_provider_config_is_refused_with_field_level_errors(client, providers):
    """Validation happens before the write, so a bad config can never reach disk.

    Otherwise the agent only finds out at the next lookup, as a confusing
    "unknown provider", long after the mistake.
    """
    response = client.post(
        "/_provider/evil", json={**_TODOIST, "token_url": "http://evil.example/token"}
    )

    assert response.status_code == 400
    assert "token_url" in response.text
    assert providers.get("evil") is None


def test_missing_required_field_is_refused(client, providers):
    response = client.post("/_provider/broken", json={"api_base": "https://x.example"})

    assert response.status_code == 400
    assert "scopes" in response.text
    assert providers.get("broken") is None


def test_a_connected_provider_cannot_be_reconfigured(client, conn, providers):
    """Rewriting token_url on a live provider would hand the refresh token away.

    The stored refresh token is the one credential that is not already reachable
    from run_bash, so this path is worth closing. Disconnect first.
    """
    conn.fetchval = AsyncMock(return_value=1)

    response = client.post("/_provider/spotify", json={**_TODOIST, "scopes": ["evil"]})

    assert response.status_code == 409
    assert "disconnect" in response.text.lower()
    assert providers.get("spotify").scopes == ["user-read-private"]


def test_removing_a_provider(client, providers):
    response = client.delete("/_provider/strava")

    assert response.status_code == 200
    assert providers.get("strava") is None
    assert providers.get("spotify") is not None


def test_a_connected_provider_cannot_be_removed(client, conn, providers):
    conn.fetchval = AsyncMock(return_value=1)

    response = client.delete("/_provider/spotify")

    assert response.status_code == 409
    assert providers.get("spotify") is not None


def test_removing_an_unknown_provider(client):
    assert client.delete("/_provider/nope").status_code == 404


def test_undecryptable_token_reports_the_cause(client, conn, upstream, monkeypatch):
    """A changed OAUTH_ENCRYPTION_KEY must not surface as an opaque 500.

    Every stored token becomes unreadable at once, and a stack trace would be read
    by the agent as "the provider is broken" rather than "the key changed".
    """
    monkeypatch.setenv("OAUTH_ENCRYPTION_KEY", "B" * 43 + "=")

    response = client.get("/spotify/v1/me")

    assert response.status_code == 503
    assert "OAUTH_ENCRYPTION_KEY" in response.text
    assert upstream.requests == []
