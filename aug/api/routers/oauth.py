"""Public OAuth 2.0 flow — the endpoints a provider's browser redirect reaches.

See ``docs/design/oauth-design.md``.  The callback is unauthenticated by
necessity, so it is inert without a live, single-use, server-side ``state``.
"""

import logging
import secrets

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from aug.config import get_settings
from aug.core.oauth.providers import ProviderConfig
from aug.core.oauth.store import claim_start_token, claim_state, create_state, save_token
from aug.utils.hushed import read_secret
from aug.utils.oauth import OAuthClient, Pkce, authorize_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth", tags=["oauth"])

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_SUCCESS_PAGE = (
    "<!doctype html><title>Connected</title><h1>Connected</h1><p>You can close this tab.</p>"
)

# The authorization code is in the callback URL.  Any external asset would send it
# onward as a Referer and any cache would retain it, so the page loads nothing and
# is never stored.
_NO_LEAK_HEADERS = {
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'",
}


@router.get("/{provider}/start")
async def oauth_start(provider: str, t: str, request: Request) -> Response:
    """Begin an authorization flow.  Unauthenticated, but gated by a minted token.

    This endpoint opens in a browser, so it cannot require ``X-API-Key``.  The
    single-use ``t`` token — minted by an authenticated action — is what stops a
    stranger from starting a flow and grafting their account onto AUG.
    """
    if not request.app.state.oauth_start_limiter.allow(_client_ip(request)):
        return _page("Too many attempts. Try again later.", 429)

    async with request.app.state.db_pool.acquire() as conn:
        start = await claim_start_token(conn, t)
        if start is None:
            logger.warning("oauth start rejected — unknown or expired start token")
            return _page("Link expired. Ask AUG for a new one.", 400)

        # The token is authoritative; the path segment only has to agree with it, so
        # that the link the user reads cannot name a different provider than it connects.
        if start["provider"] != provider:
            logger.warning("oauth start rejected — path=%s token=%s", provider, start["provider"])
            return _page("Link is not valid for this provider.", 400)

        config: ProviderConfig | None = request.app.state.oauth_providers.get(start["provider"])
        if config is None:
            logger.error("oauth start for unconfigured provider=%s", start["provider"])
            return _page(f"Provider {start['provider']!r} is not configured.", 400)

        state = secrets.token_urlsafe(32)
        pkce = Pkce.generate()
        redirect_uri = _redirect_uri(start["provider"])
        await create_state(
            conn,
            state,
            start["provider"],
            start["account"],
            pkce.verifier,
            redirect_uri,
            config.issuer,
        )

    client_id, _ = _credentials(start["provider"], config)
    url = authorize_url(config, client_id, redirect_uri, state, pkce.challenge)
    return RedirectResponse(url, 302)


@router.get("/{provider}/callback", response_class=HTMLResponse)
async def oauth_callback(
    provider: str,
    state: str,
    request: Request,
    code: str | None = None,
    error: str | None = None,
    iss: str | None = None,
) -> HTMLResponse:
    """Complete the authorization code flow and store the resulting tokens.

    ``code`` is optional because a user who declines consent is sent back with an
    ``error`` and no code — a required ``code`` would answer that deliberate choice
    with a validation error page.
    """
    # Checked before the state lookup: an unauthenticated flood must not be able to
    # buy database work, let alone an outbound request.
    if not request.app.state.oauth_callback_limiter.allow(_client_ip(request)):
        return _page("Too many attempts. Try again later.", 429)

    async with request.app.state.db_pool.acquire() as conn:
        row = await claim_state(conn, state)
        if row is None:
            logger.warning("oauth callback rejected — unknown or expired state")
            return _page("Link expired. Start again.", 400)

        if error is not None:
            logger.info("oauth declined provider=%s error=%s", provider, error)
            return _page(f"Not connected ({error}). You can close this tab.")

        if code is None:
            logger.warning("oauth callback without code or error provider=%s", provider)
            return _page("Invalid callback: no authorization code.", 400)

        # Mix-up defence (RFC 9700): the state was minted for one provider and is
        # only redeemable at that provider's callback.
        if row["provider"] != provider:
            logger.warning(
                "oauth callback rejected — provider mismatch: path=%s state=%s",
                provider,
                row["provider"],
            )
            return _page("Link is not valid for this provider.", 400)

        # RFC 9207: when the provider echoes the issuer, it must be the one we
        # sent the user to.  Absence is not fatal — most consumer providers omit it.
        if iss is not None and iss != row["issuer"]:
            logger.warning("oauth callback rejected — iss mismatch for provider=%s", provider)
            return _page("Link is not valid for this provider.", 400)

        config: ProviderConfig | None = request.app.state.oauth_providers.get(provider)
        if config is None:
            logger.error("oauth callback for unconfigured provider=%s", provider)
            return _page(f"Provider {provider!r} is not configured.", 400)

        client_id, client_secret = _credentials(provider, config)

        async with httpx.AsyncClient(
            transport=request.app.state.oauth_transport,
            timeout=_TIMEOUT,
            follow_redirects=False,
        ) as http:
            client = OAuthClient(config, client_id, client_secret, http)
            token = await client.exchange_code(code, row["redirect_uri"], row["code_verifier"])

        await save_token(conn, row["provider"], row["account"], token)

    logger.info("oauth connected provider=%s account=%s", row["provider"], row["account"])
    return _page(_SUCCESS_PAGE)


def _client_ip(request: Request) -> str:
    """The address the trusted proxy actually observed.

    With exactly one proxy hop, every X-Forwarded-For entry left of the last one was
    supplied by the caller and can be forged — taking the leftmost would let an
    attacker mint a fresh allowance per request.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


def _page(body: str, status_code: int = 200) -> HTMLResponse:
    """Render a callback page that cannot leak the code it was reached with."""
    return HTMLResponse(body, status_code=status_code, headers=_NO_LEAK_HEADERS)


def _redirect_uri(provider: str) -> str:
    """The one URI registered with the provider — exact-match per RFC 9700."""
    return f"{get_settings().base_url}/oauth/{provider}/callback"


def _credentials(provider: str, config: ProviderConfig) -> tuple[str, str]:
    """Read the provider's client credentials from hushed (or the process environment)."""
    id_var = config.client_id_env or f"{provider.upper()}_CLIENT_ID"
    secret_var = config.client_secret_env or f"{provider.upper()}_CLIENT_SECRET"
    return read_secret(id_var), read_secret(secret_var)
