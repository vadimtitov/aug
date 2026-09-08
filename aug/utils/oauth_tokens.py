"""Generic OAuth2 token storage — load, save, delete, and refresh tokens on disk.

Token files are plain JSON stored at ``/app/data/oauth_tokens/{provider}/{account}.json``.
This mirrors the Gmail token pattern but works for any OAuth2 provider.

Provider configuration (auth/token URIs, default scopes) is read dynamically from
``/app/data/oauth_providers.json``.  Client credentials are read from environment
variables using the ``{PROVIDER_UPPER}_CLIENT_ID`` / ``{PROVIDER_UPPER}_CLIENT_SECRET``
pattern — no provider-specific code anywhere.
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from aug.config import get_settings
from aug.core.dispatch import broadcast
from aug.utils.data import DATA_DIR
from aug.utils.hushed import read_secret

logger = logging.getLogger(__name__)

_TOKEN_DIR = DATA_DIR / "oauth_tokens"
_PROVIDERS_FILE = DATA_DIR / "oauth_providers.json"

# Refresh tokens within this window of expiry.
_REFRESH_WINDOW = timedelta(hours=1)
# HTTP timeout for token endpoint calls.
_HTTP_TIMEOUT = 30.0


def token_path(provider: str, account: str) -> Path:
    """Return the on-disk path for a provider/account token file."""
    return _TOKEN_DIR / provider / f"{account}.json"


def save_token(provider: str, account: str, token_data: dict) -> None:
    """Persist token data to disk as JSON with restricted permissions.

    Writes atomically (temp file + ``os.replace``) so a crash never leaves a
    corrupt half-written file.
    """
    path = token_path(provider, account)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(token_data, indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        path.chmod(0o600)
    except OSError:
        # On some filesystems chmod is a no-op; container isolation is the real boundary.
        pass
    logger.info("oauth: token saved provider=%s account=%s", provider, account)


def load_token(provider: str, account: str) -> dict | None:
    """Load a token from disk, or ``None`` if no token file exists."""
    path = token_path(provider, account)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("oauth: could not read token file %s: %s", path, e)
        return None


def delete_token(provider: str, account: str) -> None:
    """Delete a token file if it exists."""
    path = token_path(provider, account)
    if path.exists():
        path.unlink()
        logger.info("oauth: token deleted provider=%s account=%s", provider, account)


def load_provider_config(provider: str) -> dict | None:
    """Load a single provider's config from the JSON file.

    Returns ``None`` if the providers file or the provider entry is missing.
    """
    if not _PROVIDERS_FILE.exists():
        return None
    try:
        all_providers = json.loads(_PROVIDERS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("oauth: could not read providers file %s: %s", _PROVIDERS_FILE, e)
        return None
    return all_providers.get(provider)


def list_all_tokens() -> list[tuple[str, str]]:
    """Return a list of ``(provider, account)`` pairs for all stored tokens."""
    result: list[tuple[str, str]] = []
    if not _TOKEN_DIR.exists():
        return result
    for provider_dir in sorted(_TOKEN_DIR.iterdir()):
        if not provider_dir.is_dir():
            continue
        for token_file in sorted(provider_dir.glob("*.json")):
            result.append((provider_dir.name, token_file.stem))
    return result


async def get_valid_token(provider: str, account: str = "primary") -> str | None:
    """Return a valid access_token, refreshing if needed.

    Returns ``None`` if the token is dead (no file, or refresh failed).
    """
    token = load_token(provider, account)
    if token is None:
        return None

    if not _is_expired(token):
        return token.get("access_token")

    # Token is expired — try to refresh
    refresh_token = token.get("refresh_token")
    if not refresh_token:
        logger.warning(
            "oauth: token expired and no refresh_token provider=%s account=%s", provider, account
        )
        delete_token(provider, account)
        return None

    new_token = await _refresh_token(provider, account, refresh_token)
    if new_token is None:
        delete_token(provider, account)
        return None

    return new_token.get("access_token")


async def refresh_all_tokens(app) -> int:
    """Background job: refresh all tokens nearing expiry.

    Iterates all token files on disk.  For each token within ``_REFRESH_WINDOW``
    of expiry, calls the provider's token endpoint with ``grant_type=refresh_token``.
    On success, updates the token file.  On failure, deletes the token file and
    sends a Telegram notification via ``broadcast()``.

    Returns the number of tokens refreshed.
    """
    refreshed = 0
    for provider, account in list_all_tokens():
        token = load_token(provider, account)
        if token is None:
            continue

        if not _needs_refresh(token):
            continue

        refresh_token = token.get("refresh_token")
        if not refresh_token:
            logger.warning(
                "oauth: token near expiry, no refresh_token provider=%s account=%s",
                provider,
                account,
            )
            delete_token(provider, account)
            await _notify_revoked(app, provider, account)
            continue

        new_token = await _refresh_token(provider, account, refresh_token)
        if new_token is None:
            delete_token(provider, account)
            await _notify_revoked(app, provider, account)
            continue

        refreshed += 1

    if refreshed:
        logger.info("oauth: background refresh completed, refreshed=%d", refreshed)
    return refreshed


def normalize_token(response: dict, existing: dict) -> dict:
    """Normalize a token response into our standard on-disk format.

    Handles:
    - ``expires_in`` (seconds from now) → ISO 8601 ``expires_at``
    - ``expires_at`` as unix timestamp (Strava) → ISO 8601
    - Preserves ``refresh_token`` from existing token if not returned
    """
    now = datetime.now(tz=UTC)

    expires_at = None
    if "expires_in" in response:
        expires_at = now + timedelta(seconds=int(response["expires_in"]))
    elif "expires_at" in response:
        raw = response["expires_at"]
        if isinstance(raw, (int, float)):
            expires_at = datetime.fromtimestamp(raw, tz=UTC)
        elif isinstance(raw, str):
            try:
                expires_at = datetime.fromtimestamp(int(raw), tz=UTC)
            except ValueError:
                expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))

    token_data: dict = {
        "access_token": response.get("access_token", existing.get("access_token")),
        "refresh_token": response.get("refresh_token", existing.get("refresh_token")),
        "token_type": response.get("token_type", "Bearer"),
        "scope": response.get("scope", existing.get("scope", "")),
    }

    if expires_at:
        token_data["expires_at"] = expires_at.isoformat()

    return token_data


# --- Private ---


def _is_expired(token: dict) -> bool:
    """Check if a token's ``expires_at`` is in the past."""
    expires_at = token.get("expires_at")
    if not expires_at:
        return False  # No expiry info — assume valid
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    return datetime.now(tz=UTC) >= expiry


def _needs_refresh(token: dict) -> bool:
    """Check if a token is within ``_REFRESH_WINDOW`` of expiry."""
    expires_at = token.get("expires_at")
    if not expires_at:
        return False
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    return datetime.now(tz=UTC) >= expiry - _REFRESH_WINDOW


async def _refresh_token(provider: str, account: str, refresh_token: str) -> dict | None:
    """Exchange a refresh_token for a new access_token.

    Returns the updated token dict on success, ``None`` on failure.
    """
    config = load_provider_config(provider)
    if config is None:
        logger.error("oauth: no provider config for %s", provider)
        return None

    client_id = read_secret(f"{provider.upper()}_CLIENT_ID")
    client_secret = read_secret(f"{provider.upper()}_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.error("oauth: missing credentials for %s", provider)
        return None

    token_uri = config["token_uri"]
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(token_uri, data=data)
    except httpx.HTTPError as e:
        logger.warning("oauth: refresh request failed provider=%s error=%s", provider, e)
        return None

    if resp.status_code != 200:
        logger.warning(
            "oauth: refresh failed provider=%s account=%s status=%d",
            provider,
            account,
            resp.status_code,
        )
        return None

    body = resp.json()
    existing = load_token(provider, account) or {}
    updated = normalize_token(body, existing)
    save_token(provider, account, updated)
    logger.info("oauth: token refreshed provider=%s account=%s", provider, account)
    return updated


async def _notify_revoked(app, provider: str, account: str) -> None:
    """Send a Telegram notification that a token was revoked."""
    base_url = get_settings().base_url
    msg = (
        f"⚠️ {provider} token for account '{account}' is no longer valid — "
        f"access was revoked. Re-authorize at "
        f"{base_url}/auth/oauth/{provider}?account={account}"
    )
    try:
        await broadcast(app, msg)
    except Exception:
        logger.exception("oauth: failed to send revocation notification")
