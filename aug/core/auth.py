"""Telegram Mini App authentication.

Provides:
  - verify_telegram_init_data: verifies the HMAC-SHA256 signature Telegram injects
    into every Mini App launch. Returns parsed payload on success, raises ValueError
    on failure.
  - create_jwt / verify_jwt: short-lived JWT issuance and verification, signed with
    the bot token so no extra secret is needed.
"""

import hashlib
import hmac
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qsl

import jwt

_JWT_ALGORITHM = "HS256"
_JWT_EXPIRES_SECONDS = 86400  # 24 hours
_INIT_DATA_MAX_AGE_SECONDS = 86400  # 24 hours (Telegram recommendation)


def verify_telegram_init_data(init_data: str, bot_token: str) -> dict:
    """Verify a Telegram Mini App initData string.

    Args:
        init_data: Raw URL-encoded initData string from Telegram.
        bot_token: Telegram bot token used as the HMAC key.

    Raises:
        ValueError: if the hash is missing, invalid, or the data is expired.

    Returns:
        Parsed key-value dict of the initData fields (excluding hash).
    """
    params = dict(parse_qsl(init_data, keep_blank_values=True))

    received_hash = params.pop("hash", None)
    if not received_hash:
        raise ValueError("Hash missing from init data")

    # `signature` (Ed25519, Bot API 7.7+) is not part of the HMAC data check string.
    params.pop("signature", None)

    auth_date = int(params.get("auth_date", 0))
    if time.time() - auth_date > _INIT_DATA_MAX_AGE_SECONDS:
        raise ValueError("Init data expired")

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(params.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), digestmod=hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), digestmod=hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        raise ValueError("Invalid hash — init data may have been tampered with")

    return params


def create_jwt(payload: dict, secret: str) -> str:
    """Issue a signed JWT valid for 24 hours."""
    claims = {
        **payload,
        "exp": datetime.now(UTC) + timedelta(seconds=_JWT_EXPIRES_SECONDS),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(claims, secret, algorithm=_JWT_ALGORITHM)


def verify_jwt(token: str, secret: str) -> dict:
    """Decode and verify a JWT. Raises jwt.PyJWTError on failure."""
    return jwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
