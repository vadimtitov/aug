"""API authentication dependency.

Accepts either:
  - X-API-Key header (shared secret, used by server-side integrations)
  - Authorization: Bearer <jwt> (issued by POST /auth/telegram, used by the Mini App)

Usage — apply at router level:

    router = APIRouter(dependencies=[Depends(require_api_key)])
"""

import hmac
import logging

import jwt
from fastapi import HTTPException, Request, status

from aug.config import get_settings
from aug.core.auth import verify_jwt

logger = logging.getLogger(__name__)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency — raises 401 if the request is not authenticated."""
    settings = get_settings()

    # --- X-API-Key ---
    api_key = request.headers.get("X-API-Key")
    if api_key:
        if hmac.compare_digest(api_key, settings.API_KEY):
            return
        logger.warning("auth_failed method=api_key reason=invalid")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    # --- Bearer JWT ---
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")
        bot_token = settings.TELEGRAM_BOT_TOKEN
        if not bot_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
            )
        try:
            verify_jwt(token, bot_token)
            return
        except jwt.PyJWTError as exc:
            logger.warning("auth_failed method=jwt reason=%s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
            ) from exc

    logger.warning("auth_failed method=none")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_ws_credential(token: str) -> bool:
    """Validate a Mini App JWT presented over a WebSocket subprotocol.

    Browser WebSocket clients cannot set the Authorization header, so the Mini App
    offers its JWT as a ``Sec-WebSocket-Protocol`` value. That keeps the credential
    out of the URL — and therefore out of access logs, proxy logs, and browser
    history — unlike a ``?token=`` query param.

    Only the short-lived Mini App JWT is accepted here, never the permanent shared
    API key: a credential that can be captured from a screencast surface should be
    one that expires on its own. Returns ``True`` if valid (never raises).
    """
    if not token:
        return False
    bot_token = get_settings().TELEGRAM_BOT_TOKEN
    if not bot_token:
        return False
    try:
        verify_jwt(token, bot_token)
        return True
    except jwt.PyJWTError as exc:
        logger.warning("ws_auth_failed reason=%s", exc)
        return False
