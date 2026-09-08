"""OAuth 2.0 protocol client — how to talk to an authorization server.

Knows nothing about storage or HTTP routing.  The caller supplies the
``httpx.AsyncClient`` so that timeouts, transports and redirect policy are set in
one place and tests can exercise the real request-building code.
"""

import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from aug.core.oauth.providers import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pkce:
    """An RFC 7636 verifier and its S256 challenge."""

    verifier: str
    challenge: str

    @classmethod
    def generate(cls) -> "Pkce":
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode()).digest()
        return cls(verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode())


@dataclass(frozen=True)
class TokenResponse:
    """A token endpoint's answer, normalised."""

    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str
    expires_at: datetime | None


class OAuthClient:
    """Authorization-code exchange and refresh for a single provider."""

    def __init__(
        self,
        config: ProviderConfig,
        client_id: str,
        client_secret: str,
        http: httpx.AsyncClient,
    ) -> None:
        self.config = config
        self.client_id = client_id
        self.client_secret = client_secret
        self.http = http

    async def exchange_code(
        self, code: str, redirect_uri: str, code_verifier: str
    ) -> TokenResponse:
        """Trade an authorization code for tokens (RFC 6749 §4.1.3, with PKCE)."""
        return await self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            }
        )

    async def refresh(self, refresh_token: str) -> TokenResponse:
        """Exchange a refresh token for a new access token (RFC 6749 §6).

        Providers that rotate return a new refresh token here; the caller must
        persist it, because the one just used is now dead.
        """
        return await self._token_request(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        )

    async def revoke(self, token: str) -> None:
        """Ask the provider to invalidate a token (RFC 7009).

        Raises on failure: the caller decides what to do, and must not report a
        revocation that did not happen.
        """
        response = await self.http.post(
            self.config.revoke_url,
            data={"token": token, "client_id": self.client_id, "client_secret": self.client_secret},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

    async def _token_request(self, form: dict[str, str]) -> TokenResponse:
        """POST to the token endpoint with the configured client authentication."""
        auth = None
        if self.config.token_auth_method == "basic":
            auth = (self.client_id, self.client_secret)
        else:
            form = form | {"client_id": self.client_id, "client_secret": self.client_secret}

        response = await self.http.post(
            self.config.token_url,
            data=form,
            auth=auth,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()

        expires_in = payload.get("expires_in")
        return TokenResponse(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            token_type=payload.get("token_type", "Bearer"),
            scope=payload.get("scope", ""),
            # `expires_in: 0` means already expired, not "no expiry" — hence `is not None`.
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=int(expires_in))
                if expires_in is not None
                else None
            ),
        )


def authorize_url(
    config: ProviderConfig, client_id: str, redirect_uri: str, state: str, challenge: str
) -> str:
    """Build the URL the user's browser is redirected to for consent.

    A plain function, not an ``OAuthClient`` method: nothing is sent, so requiring
    a configured HTTP client here would be a lie about what the call does.
    """
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": config.scope_separator.join(config.scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        **config.extra_authorize_params,
    }
    return str(httpx.URL(config.authorize_url, params=params))
