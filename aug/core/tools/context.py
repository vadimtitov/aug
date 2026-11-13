"""Ambient LangGraph run context — thread_id / interface / ToolContext for the tool call
in progress."""

from langgraph.config import get_config

from aug.core.harness.tool_context import TOOL_CONTEXT_CONFIG_KEY, ToolContext


def current_thread_id() -> str:
    """The LangGraph thread_id for the current tool call, or "" outside a run."""
    return _configurable("thread_id")


def current_interface() -> str:
    """The interface name (e.g. "telegram") for the current tool call, or "" outside a run."""
    return _configurable("interface")


def current_tool_context() -> ToolContext | None:
    """The running turn's ToolContext, or None outside a turn (e.g. a unit test)."""
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return None
    return configurable.get(TOOL_CONTEXT_CONFIG_KEY)


def _configurable(key: str) -> str:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ""
    return configurable.get(key) or ""
