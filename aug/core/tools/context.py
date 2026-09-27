"""Ambient LangGraph run context — thread_id / interface for the tool call in progress."""

from langgraph.config import get_config


def current_thread_id() -> str:
    """The LangGraph thread_id for the current tool call, or "" outside a run."""
    return _configurable("thread_id")


def current_interface() -> str:
    """The interface name (e.g. "telegram") for the current tool call, or "" outside a run."""
    return _configurable("interface")


def _configurable(key: str) -> str:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ""
    return configurable.get(key) or ""
