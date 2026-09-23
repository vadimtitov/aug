"""Tests for the v12_claude / MCP-tool wiring in aug/core/registry.py.

V11 agents are immutable — this only exercises the new v12_claude factory path
(configure_mcp_tools / v12_base_tool_names) added for MCP support.
"""

from langchain_core.tools import tool

from aug.core.registry import (
    _V11_CLAUDE_TOOLS,
    configure_mcp_tools,
    get_agent,
    list_agents,
    v12_base_tool_names,
)


@tool
def _fake_mcp_tool() -> str:
    """A fake MCP-sourced tool for tests."""
    return "ok"


def test_v12_claude_is_registered():
    assert "v12_claude" in list_agents()


def test_v12_claude_starts_with_no_mcp_tools_by_default():
    """Before configure_mcp_tools() runs (e.g. at import time, before app startup),
    v12_claude degrades gracefully to just its base tool set."""
    configure_mcp_tools([])
    agent = get_agent("v12_claude")
    names = {t.name for t in agent.tools}
    assert "_fake_mcp_tool" not in names
    assert "search_mcp_servers" in names
    assert "install_mcp_server" in names
    assert "list_mcp_servers" in names
    assert "remove_mcp_server" in names


def test_v11_claude_untouched_by_v12_tool_names():
    """v12_claude tacks MCP management tools onto v11's set — v11 itself never
    gains them (immutability rule: existing agent versions aren't modified)."""
    v11 = get_agent("v11_claude")
    v11_names = {t.name for t in v11.tools}
    assert "search_mcp_servers" not in v11_names
    assert "install_mcp_server" not in v11_names


def test_configure_mcp_tools_appends_live_mcp_tools():
    try:
        configure_mcp_tools([_fake_mcp_tool])
        agent = get_agent("v12_claude")
        names = {t.name for t in agent.tools}
        assert "_fake_mcp_tool" in names
    finally:
        configure_mcp_tools([])  # restore default for other tests


def test_v12_base_tool_names_excludes_mcp_sourced_tools():
    names = v12_base_tool_names()
    assert "search_mcp_servers" in names
    assert "note" in names  # inherited from v11
    assert "_fake_mcp_tool" not in names


def test_v11_claude_tools_list_is_a_strict_prefix_of_v12_base():
    """Documents the intended relationship: v12 = v11 + MCP management tools."""
    v11_names = {t.name for t in _V11_CLAUDE_TOOLS}
    assert v11_names.issubset(v12_base_tool_names())
