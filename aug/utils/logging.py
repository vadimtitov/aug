"""Structured logging setup.

- DEBUG=true  → human-readable format (development)
- DEBUG=false → JSON format (production / log aggregators)

Call ``configure_logging()`` once at application startup before any loggers
are created.
"""

import json
import logging
import sys
from datetime import UTC, datetime


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(debug: bool = False) -> None:
    """Configure root logger.  Call once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    if debug:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    else:
        handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Silence noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
