"""Ambient LangGraph run context — thread / interface / user of the tool call in progress."""

from langgraph.config import get_config


def current_thread_id() -> str:
    """The LangGraph thread_id for the current tool call, or "" outside a run."""
    return _configurable("thread_id")


def current_interface() -> str:
    """The interface name (e.g. "telegram") for the current tool call, or "" outside a run."""
    return _configurable("interface")


def current_user_id() -> str:
    """The person behind the current run: their platform user id, falling back to the
    delivery target (sender_id) when the interface has no separate user identity."""
    return _configurable("user_id") or _configurable("sender_id")


def _configurable(key: str) -> str:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ""
    return configurable.get(key) or ""
