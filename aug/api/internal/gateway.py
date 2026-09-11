"""Loopback token gateway — 127.0.0.1:8799.

Lets the agent *use* an OAuth credential without being able to *read* it.  The
agent calls ``localhost:8799/{provider}/{path}``; the gateway decrypts the stored
token, attaches it, and forwards the request to the provider's pinned ``api_base``.

Bound to loopback so that a credential-attaching endpoint is structurally
unreachable from the internet, rather than being defended by a reverse-proxy rule.
"""

import asyncio
import logging
import secrets
import socket

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from aug.config import get_settings
from aug.core.oauth.providers import ProviderConfig
from aug.core.oauth.refresh import TokenRefresher
from aug.core.oauth.store import (
    TokenUnreadable,
    count_connections,
    create_start_token,
    delete_token,
    list_connections,
    load_token,
)
from aug.utils.hushed import read_secret
from aug.utils.oauth import OAuthClient

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_PORT = 8799

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Hop-by-hop and identity-bearing headers never survive the hop: the client's own
# Authorization must not shadow the one we attach, and Host belongs to the upstream.
_STRIP_REQUEST_HEADERS = {"authorization", "host", "cookie", "content-length"}


async def serve_gateway(state) -> None:
    """Run the gateway for the life of the application.

    It listens unconditionally, even with nothing connected: a refused connection
    must mean "the gateway is down", never "nothing is connected" — the agent
    cannot tell those apart, and would report a broken gateway as a dead provider.

    Failure here is not fatal. AUG boots and runs without OAuth; only
    OAuth-authenticated calls stop working, and they say so.
    """
    # Bind before handing the socket to uvicorn: uvicorn exits the process when it
    # cannot bind, and a port conflict must not be able to take AUG down with it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((_HOST, _PORT))
    except OSError:
        sock.close()
        logger.exception(
            "oauth gateway could not bind %s:%d — provider calls will fail", _HOST, _PORT
        )
        return

    config = uvicorn.Config(create_gateway_app(state), log_level="warning", access_log=False)
    try:
        await uvicorn.Server(config).serve(sockets=[sock])
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("oauth gateway stopped — authenticated provider calls will fail")
    finally:
        sock.close()


def create_gateway_app(state) -> FastAPI:
    """Build the gateway app.

    ``state`` is the main application's ``app.state`` — the gateway shares its pool,
    provider registry and HTTP transport rather than keeping copies that could drift.
    """
    app = FastAPI(title="AUG OAuth gateway", docs_url=None, redoc_url=None)
    router = APIRouter()
    refresher = TokenRefresher(state)

    @router.get("/")
    async def status() -> dict:
        """What is connected right now.

        The gateway listens unconditionally so this always answers: a refused
        connection means the gateway is down, never "nothing is connected".
        """
        async with state.db_pool.acquire() as conn:
            rows = await list_connections(conn)
        return {
            "connected": [
                {
                    "provider": row["provider"],
                    "account": row["account"],
                    "healthy": not row["needs_reauth"],
                }
                for row in rows
            ],
            "configured": sorted(state.oauth_providers),
        }

    @router.post("/_provider/{provider}")
    async def save_provider(provider: str, request: Request) -> Response:
        """Validate and store one provider's configuration.

        The only writer of the config file. Config is validated before it can reach
        disk, so a malformed entry cannot be discovered later as a baffling
        "unknown provider", and the file stays parseable for every other provider.
        """
        if await _connection_count(state, provider):
            return Response(
                f"{provider} is connected. Reconfiguring a live provider could redirect "
                f"its refresh token — disconnect it first "
                f"(DELETE /{provider}), then save the new config.\n",
                status_code=409,
            )

        try:
            config = state.oauth_providers.save(provider, await request.json())
        except ValidationError as exc:
            return Response(_validation_errors(exc), status_code=400)
        except ValueError:
            return Response("Body must be a JSON object.\n", status_code=400)

        return JSONResponse({"provider": provider, "config": config.model_dump(exclude_none=True)})

    @router.delete("/_provider/{provider}")
    async def remove_provider(provider: str) -> Response:
        """Forget a provider's configuration.  Must be disconnected first."""
        if await _connection_count(state, provider):
            return Response(
                f"{provider} is connected. Disconnect it first (DELETE /{provider}).\n",
                status_code=409,
            )
        if not state.oauth_providers.remove(provider):
            return _unknown_provider(provider, state.oauth_providers)
        return Response(f"Removed {provider} configuration.\n")

    @router.post("/_link/{provider}")
    async def mint_link(provider: str, request: Request) -> Response:
        """Mint a single-use start link for the user to open in a browser.

        ``/start`` cannot require an API key — it opens in a browser — so this is
        what stops a stranger from beginning a flow and grafting their own account.
        Minting is safe to expose here because the gateway is loopback-only.
        """
        if provider not in state.oauth_providers:
            return _unknown_provider(provider, state.oauth_providers)

        account = request.headers.get("X-Aug-Account", "primary")
        token = secrets.token_urlsafe(32)
        async with state.db_pool.acquire() as conn:
            await create_start_token(conn, token, provider, account)

        base = get_settings().base_url
        logger.info("oauth start link minted provider=%s account=%s", provider, account)
        return JSONResponse({"url": f"{base}/oauth/{provider}/start?t={token}"})

    @router.delete("/{provider}")
    async def disconnect(provider: str, request: Request) -> Response:
        """Revoke where the provider supports it, then forget the token locally.

        Deleting the row is *not* revocation — the grant stays live at the provider
        and the refresh token keeps working. Saying "disconnected" when only the
        local row went would be exactly the kind of false success the tool standard
        forbids, so the answer states which of the two actually happened.
        """
        config: ProviderConfig | None = state.oauth_providers.get(provider)
        if config is None:
            return _unknown_provider(provider, state.oauth_providers)

        account = request.headers.get("X-Aug-Account", "primary")
        async with state.db_pool.acquire() as conn:
            token = await load_token(conn, provider, account)
            if token is None:
                return Response(f"{provider}/{account} is not connected.\n", status_code=404)

            revoked = await _revoke(state, config, provider, token.access_token)
            # Delete even when revocation failed, or a provider outage would leave a
            # row nothing can remove.
            await delete_token(conn, provider, account)

        if revoked:
            return Response(f"Disconnected {provider}/{account} and revoked access.\n")
        return Response(
            f"Deleted {provider}/{account} from AUG. The app's access was NOT revoked — "
            f"remove it in your {provider} account settings.\n"
        )

    @router.api_route(
        "/{provider}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy(provider: str, path: str, request: Request) -> Response:
        """Forward an agent request to the provider, authenticated."""
        config: ProviderConfig | None = state.oauth_providers.get(provider)
        if config is None:
            return _unknown_provider(provider, state.oauth_providers)

        account = request.headers.get("X-Aug-Account", "primary")
        try:
            token = await refresher.valid_token(provider, account)
        except TokenUnreadable as exc:
            logger.error("gateway cannot read token provider=%s: %s", provider, exc)
            return Response(f"{exc}\n", status_code=503)

        if token is None:
            return Response(
                f"{provider}/{account} is not connected. Ask AUG to connect {provider} first.\n",
                status_code=401,
            )

        if token.needs_reauth:
            return Response(
                f"{provider}/{account} requires re-authorization "
                f"(refresh failed: {token.last_error}). Ask AUG to reconnect {provider}.\n",
                status_code=503,
            )

        url = _upstream_url(config, path, request.url.query)
        if url is None:
            logger.warning("gateway blocked off-host path provider=%s path=%r", provider, path)
            return Response(
                f"Path {path!r} would leave {config.api_base}. Refused.\n", status_code=400
            )

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _STRIP_REQUEST_HEADERS
        }
        headers["Authorization"] = f"{token.token_type} {token.access_token}"

        async with httpx.AsyncClient(
            transport=state.oauth_transport, timeout=_TIMEOUT, follow_redirects=False
        ) as http:
            upstream = await http.request(
                request.method, url, headers=headers, content=await request.body()
            )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers={
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
            },
        )

    app.include_router(router)
    return app


async def _connection_count(state, provider: str) -> int:
    """How many accounts are live for a provider — the guard on config changes."""
    async with state.db_pool.acquire() as conn:
        return await count_connections(conn, provider)


def _validation_errors(exc: ValidationError) -> str:
    """Field-level errors the agent can act on, not a stack trace."""
    lines = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return "Invalid provider config:\n" + "\n".join(lines) + "\n"


async def _revoke(state, config: ProviderConfig, provider: str, access_token: str) -> bool:
    """Best-effort revocation.  False when unsupported or the provider refused."""
    if not config.revoke_url:
        return False
    client_id = read_secret(config.client_id_env or f"{provider.upper()}_CLIENT_ID")
    secret = read_secret(config.client_secret_env or f"{provider.upper()}_CLIENT_SECRET")
    try:
        async with httpx.AsyncClient(
            transport=state.oauth_transport, timeout=_TIMEOUT, follow_redirects=False
        ) as http:
            await OAuthClient(config, client_id, secret, http).revoke(access_token)
        return True
    except httpx.HTTPError:
        logger.exception("oauth revocation failed provider=%s — deleting locally anyway", provider)
        return False


def _unknown_provider(provider: str, providers: dict | None = None) -> Response:
    """Name the providers that do exist — a bare 404 tells the agent nothing."""
    known = ", ".join(sorted(providers or {})) or "(none configured)"
    return Response(
        f"Unknown provider {provider!r}. Configured providers: {known}\n", status_code=404
    )


def _upstream_url(config: ProviderConfig, path: str, query: str) -> httpx.URL | None:
    """Resolve an agent-supplied path against the provider's pinned base.

    Returns None if the result would leave that host.  A protocol-relative path
    (``//evil.example/x``) or an absolute URL would otherwise carry the token to a
    host of the caller's choosing, which is the one thing the gateway exists to stop.
    """
    base = httpx.URL(config.api_base)
    candidate = base.join("/" + path.lstrip("/"))
    if (candidate.scheme, candidate.host, candidate.port) != (base.scheme, base.host, base.port):
        return None
    return candidate.copy_with(query=query.encode() or None)
