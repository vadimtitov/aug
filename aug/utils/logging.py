"""Structured logging setup.

- DEBUG=true  → human-readable format (development)
- DEBUG=false → JSON format (production / log aggregators)

Call ``configure_logging()`` once at application startup before any loggers
are created.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from langchain_core.messages import AIMessage

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_thread_id: ContextVar[str] = ContextVar("thread_id", default="-")


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async context."""
    _correlation_id.set(cid)


def set_thread_id(tid: str) -> None:
    """Set the thread (conversation) ID for the current async context."""
    _thread_id.set(tid)


class _ContextFilter(logging.Filter):
    """Inject cid and tid into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()  # type: ignore[attr-defined]
        record.thread_id = _thread_id.get()  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "cid": _correlation_id.get(),
            "tid": _thread_id.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_token_logger = logging.getLogger("aug.tokens")


def log_token_usage(response: AIMessage) -> None:
    """Log prompt/completion/total token counts from an LLM response at DEBUG level."""
    usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage")
    if usage:
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        completion = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)
        _token_logger.debug(
            "token_usage prompt=%d completion=%d total=%d", prompt, completion, total
        )


def configure_logging(debug: bool = False) -> None:
    """Configure root logger.  Call once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if debug:
        fmt = (
            "%(asctime)s [%(correlation_id)s/%(thread_id)s] %(levelname)-8s %(name)s — %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt))
    else:
        handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Silence noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
