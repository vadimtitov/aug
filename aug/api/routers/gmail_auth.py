"""Gmail OAuth 2.0 flow.

Endpoints:
  GET /auth/gmail?account=primary   → redirects to Google consent screen
  GET /auth/gmail/callback          → exchanges code for token, saves to disk
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow

from aug.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/gmail", tags=["gmail-auth"])

_SCOPES = ["https://mail.google.com/"]
_TOKEN_DIR = Path("/app/data/gmail_tokens")


def _redirect_uri() -> str:
    return f"{get_settings().base_url}/auth/gmail/callback"


def _client_config() -> dict:
    client_id = get_settings().GMAIL_CLIENT_ID
    client_secret = get_settings().GMAIL_CLIENT_SECRET
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=503,
            detail="Gmail OAuth is not configured: GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET missing.",
        )
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [_redirect_uri()],
        }
    }


def _flow(state: str | None = None) -> Flow:
    return Flow.from_client_config(
        _client_config(),
        scopes=_SCOPES,
        redirect_uri=_redirect_uri(),
        state=state,
    )


def token_path(account: str) -> Path:
    return _TOKEN_DIR / f"{account}.json"


def save_token(account: str, token: dict) -> None:
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token_path(account).write_text(json.dumps(token))
    logger.info("gmail_auth: token saved for account=%r", account)


def load_token(account: str) -> dict | None:
    path = token_path(account)
    if not path.exists():
        return None
    return json.loads(path.read_text())


@router.get("")
async def gmail_auth_start(account: str = "primary") -> RedirectResponse:
    """Kick off the OAuth flow. Visit this URL in a browser."""
    flow = _flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=account,
        include_granted_scopes="true",
    )
    logger.info("gmail_auth: redirecting account=%r to Google consent", account)
    return RedirectResponse(auth_url)


@router.get("/callback")
async def gmail_auth_callback(code: str, state: str = "primary") -> dict:
    """Google redirects here after user grants consent."""
    account = state
    flow = _flow(state=account)
    try:
        flow.fetch_token(code=code)
    except Exception as e:
        logger.error("gmail_auth: token exchange failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {e}") from e

    creds = flow.credentials
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or _SCOPES),
    }
    save_token(account, token_data)
    return {
        "status": "ok",
        "account": account,
        "message": f"Gmail connected for account '{account}'.",
    }
