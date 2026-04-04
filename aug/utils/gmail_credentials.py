"""Gmail OAuth token storage — load and save tokens to disk."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_DIR = Path("/app/data/gmail_tokens")


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
