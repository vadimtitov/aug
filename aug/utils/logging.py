"""Logging setup.

Call ``configure_logging()`` once at application startup.

- DEBUG=false (default) — INFO level, concise human-readable format
- DEBUG=true            — DEBUG level, includes correlation/thread IDs
"""

import logging
import sys
from contextvars import ContextVar

from langchain_core.messages import AIMessage

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_thread_id: ContextVar[str] = ContextVar("thread_id", default="-")


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def set_thread_id(tid: str) -> None:
    _thread_id.set(tid)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()  # type: ignore[attr-defined]
        record.thread_id = _thread_id.get()  # type: ignore[attr-defined]
        return True


class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()


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
    """Configure root logger. Call once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())

    if debug:
        fmt = "%(asctime)s [%(levelname)s] %(correlation_id)s/%(thread_id)s %(name)s — %(message)s"
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())
