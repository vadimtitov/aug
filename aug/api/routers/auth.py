"""Auth router — POST /auth/telegram."""

import json
import logging
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from aug.config import get_settings
from aug.core.auth import create_jwt, verify_telegram_init_data

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


class TelegramAuthRequest(BaseModel):
    init_data: str


class TelegramAuthResponse(BaseModel):
    token: str


@router.post("/auth/telegram", response_model=TelegramAuthResponse)
async def auth_telegram(body: TelegramAuthRequest) -> TelegramAuthResponse:
    """Exchange Telegram initData for a JWT.

    The client sends the raw initData string injected by Telegram on Mini App launch.
    The backend verifies the HMAC-SHA256 signature using the bot token, then issues
    a short-lived JWT the client uses for all subsequent requests.
    """
    settings = get_settings()
    bot_token = settings.TELEGRAM_BOT_TOKEN
    logger.info(
        "telegram_auth_attempt bypass=%s bot_token_set=%s init_data_len=%d init_data=%r",
        settings.DEV_AUTH_BYPASS,
        bool(bot_token),
        len(body.init_data),
        body.init_data,
    )
    if not bot_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot token not configured.",
        )

    if settings.DEV_AUTH_BYPASS:
        # Skip HMAC verification in local dev. DEV_AUTH_BYPASS must never be True
        # in production — it accepts any initData string without signature checking.
        logger.warning("telegram_auth dev_auth_bypass=True — skipping verification")
        payload = dict(parse_qsl(body.init_data, keep_blank_values=True))
        payload.pop("hash", None)
    else:
        try:
            payload = verify_telegram_init_data(body.init_data, bot_token)
        except ValueError as exc:
            _params = dict(parse_qsl(body.init_data, keep_blank_values=True))
            logger.warning(
                "telegram_auth_failed reason=%s auth_date=%s hash_present=%s fields=%s",
                exc,
                _params.get("auth_date", "MISSING"),
                "hash" in _params,
                list(_params.keys()),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Telegram init data.",
            ) from exc

    try:
        user_id = str(json.loads(payload.get("user", "{}")).get("id", ""))
    except (json.JSONDecodeError, AttributeError):
        user_id = ""

    if not settings.DEV_AUTH_BYPASS:
        allowed = settings.allowed_chat_ids
        if allowed and user_id not in {str(i) for i in allowed}:
            logger.warning("telegram_auth_rejected user_id=%s reason=not_in_allowlist", user_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied.",
            )

    token = create_jwt({"sub": user_id, "src": "telegram"}, secret=bot_token)
    return TelegramAuthResponse(token=token)
