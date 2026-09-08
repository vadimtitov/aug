"""Encrypted token persistence.

Tokens live in Postgres rather than in hushed because refresh must be
transactional: providers that rotate the refresh token invalidate the old one, so
a lost update permanently disconnects the account.

Functions take an ``asyncpg`` connection so the caller owns the transaction.
"""

import base64
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from aug.utils.oauth import TokenResponse

logger = logging.getLogger(__name__)

_NONCE_BYTES = 12

_CREATE_START_TOKEN = """
INSERT INTO oauth_start_tokens (token, provider, account, expires_at)
VALUES ($1, $2, $3, NOW() + INTERVAL '10 minutes')
"""

_CLAIM_START_TOKEN = """
DELETE FROM oauth_start_tokens
WHERE token = $1 AND expires_at > NOW()
RETURNING provider, account
"""

_CREATE_STATE = """
INSERT INTO oauth_states (state, provider, account, code_verifier, redirect_uri, issuer, expires_at)
VALUES ($1, $2, $3, $4, $5, $6, NOW() + INTERVAL '10 minutes')
"""

_CLAIM_STATE = """
DELETE FROM oauth_states
WHERE state = $1 AND expires_at > NOW()
RETURNING provider, account, code_verifier, redirect_uri, issuer
"""

_DELETE_TOKEN = """
DELETE FROM oauth_tokens WHERE provider = $1 AND account = $2
"""

_MARK_NEEDS_REAUTH = """
UPDATE oauth_tokens SET needs_reauth = TRUE, last_error = $3, updated_at = NOW()
WHERE provider = $1 AND account = $2
"""

_COUNT_CONNECTIONS = """
SELECT COUNT(*) FROM oauth_tokens WHERE provider = $1
"""

_LIST_CONNECTIONS = """
SELECT provider, account, needs_reauth FROM oauth_tokens ORDER BY provider, account
"""

_LOAD_TOKEN = """
SELECT provider, account, access_token_enc, refresh_token_enc, token_type, scopes,
       expires_at, needs_reauth, last_error
FROM oauth_tokens
WHERE provider = $1 AND account = $2
"""

_UPSERT_TOKEN = """
INSERT INTO oauth_tokens
    (provider, account, access_token_enc, refresh_token_enc, token_type, scopes, expires_at,
     needs_reauth, last_error, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, FALSE, NULL, NOW())
ON CONFLICT (provider, account) DO UPDATE SET
    access_token_enc  = EXCLUDED.access_token_enc,
    refresh_token_enc = COALESCE(EXCLUDED.refresh_token_enc, oauth_tokens.refresh_token_enc),
    token_type        = EXCLUDED.token_type,
    scopes            = EXCLUDED.scopes,
    expires_at        = EXCLUDED.expires_at,
    needs_reauth      = FALSE,
    last_error        = NULL,
    updated_at        = NOW()
"""


def encrypt(plaintext: str, provider: str, account: str) -> bytes:
    """AES-256-GCM encrypt, bound to (provider, account) as additional data.

    The AAD binding means a row copied over another row fails to decrypt rather
    than silently authenticating as the wrong account.
    """
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_key()).encrypt(nonce, plaintext.encode(), _aad(provider, account))
    return nonce + ciphertext


def decrypt(blob: bytes, provider: str, account: str) -> str:
    """Reverse of ``encrypt``.  Raises ``InvalidTag`` if the AAD does not match."""
    nonce, ciphertext = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    return AESGCM(_key()).decrypt(nonce, ciphertext, _aad(provider, account)).decode()


class TokenUnreadable(Exception):
    """A stored token cannot be decrypted — almost always a changed encryption key."""


@dataclass(frozen=True)
class StoredToken:
    """A decrypted token row.  Never logged, never returned to the agent."""

    provider: str
    account: str
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_at: datetime | None
    needs_reauth: bool
    last_error: str | None


async def count_connections(conn: asyncpg.Connection, provider: str) -> int:
    """How many accounts are connected for a provider, across all account names."""
    return await conn.fetchval(_COUNT_CONNECTIONS, provider)


async def list_connections(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    """Every connected (provider, account), with its health."""
    return await conn.fetch(_LIST_CONNECTIONS)


async def load_token(conn: asyncpg.Connection, provider: str, account: str) -> StoredToken | None:
    """Read and decrypt a stored token, or None if the account is not connected."""
    row = await conn.fetchrow(_LOAD_TOKEN, provider, account)
    if row is None:
        return None
    try:
        return _decrypt_row(row, provider, account)
    except InvalidTag as exc:
        raise TokenUnreadable(
            f"{provider}/{account} cannot be decrypted — OAUTH_ENCRYPTION_KEY has "
            f"changed since it was stored. Reconnect the provider."
        ) from exc


def _decrypt_row(row, provider: str, account: str) -> StoredToken:
    return StoredToken(
        provider=provider,
        account=account,
        access_token=decrypt(row["access_token_enc"], provider, account),
        refresh_token=(
            decrypt(row["refresh_token_enc"], provider, account)
            if row["refresh_token_enc"]
            else None
        ),
        token_type=row["token_type"],
        expires_at=row["expires_at"],
        needs_reauth=row["needs_reauth"],
        last_error=row["last_error"],
    )


async def create_start_token(
    conn: asyncpg.Connection, token: str, provider: str, account: str
) -> None:
    """Record a start-link token.  Short-lived and single-use — see claim_start_token."""
    await conn.execute(_CREATE_START_TOKEN, token, provider, account)


async def claim_start_token(conn: asyncpg.Connection, token: str) -> asyncpg.Record | None:
    """Consume a start-link token, or return None if there is no live one.

    Start links are minted by an authenticated action, so an unauthenticated caller
    cannot begin a flow and graft their own provider account onto AUG.
    """
    return await conn.fetchrow(_CLAIM_START_TOKEN, token)


async def create_state(
    conn: asyncpg.Connection,
    state: str,
    provider: str,
    account: str,
    code_verifier: str,
    redirect_uri: str,
    issuer: str | None,
) -> None:
    """Record a pending authorization, to be claimed once by the callback."""
    await conn.execute(_CREATE_STATE, state, provider, account, code_verifier, redirect_uri, issuer)


async def claim_state(conn: asyncpg.Connection, state: str) -> asyncpg.Record | None:
    """Consume a pending authorization state, or return None if there is no live one.

    The DELETE ... RETURNING is what makes the state single-use and race-free: two
    concurrent callbacks with the same state cannot both get a row.
    """
    return await conn.fetchrow(_CLAIM_STATE, state)


async def save_token(
    conn: asyncpg.Connection, provider: str, account: str, token: TokenResponse
) -> None:
    """Persist a freshly issued or refreshed token, encrypted."""
    await conn.execute(
        _UPSERT_TOKEN,
        provider,
        account,
        encrypt(token.access_token, provider, account),
        encrypt(token.refresh_token, provider, account) if token.refresh_token else None,
        token.token_type,
        token.scope,
        token.expires_at,
    )


async def delete_token(conn: asyncpg.Connection, provider: str, account: str) -> None:
    """Forget a connection locally.  Revocation, where supported, is a separate step."""
    await conn.execute(_DELETE_TOKEN, provider, account)


async def mark_needs_reauth(
    conn: asyncpg.Connection, provider: str, account: str, error: str
) -> None:
    """Flag a dead grant.  The row is kept: it is what makes the failure explainable."""
    await conn.execute(_MARK_NEEDS_REAUTH, provider, account, error)


def _key() -> bytes:
    """Decode OAUTH_ENCRYPTION_KEY, failing loudly if it is missing or malformed."""
    raw = os.environ.get("OAUTH_ENCRYPTION_KEY", "")
    if not raw:
        raise RuntimeError("OAUTH_ENCRYPTION_KEY is not set — OAuth tokens cannot be stored")
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise RuntimeError(f"OAUTH_ENCRYPTION_KEY must decode to 32 bytes, got {len(key)}")
    return key


def _aad(provider: str, account: str) -> bytes:
    return f"{provider}|{account}".encode()
