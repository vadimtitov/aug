"""Tests for the v12 / MCP-tool wiring in aug/core/registry.py.

V11 agents are immutable — this only exercises the new v12 factory path
(configure_mcp_tools / v12_base_tool_names) added for MCP support.
"""

import pytest
from langchain_core.tools import tool

from aug.core.agents.chat_agent import AugAgent
from aug.core.registry import (
    _V11_CLAUDE_TOOLS,
    _V11_GLM5_TOOLS,
    _V11_KIMI_TOOLS,
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


def test_v12_base_is_v11_without_browser_plus_mcp_management_tools():
    """Documents the intended relationship: v12 = v11 - browser + MCP management tools."""
    v11_names = {t.name for t in _V11_CLAUDE_TOOLS}
    assert v12_base_tool_names() == (v11_names - {"browser"}) | {
        "search_mcp_servers",
        "install_mcp_server",
        "list_mcp_servers",
        "remove_mcp_server",
    }


def test_v12_claude_has_no_browser_tool_but_v11_keeps_it():
    configure_mcp_tools([])
    assert "browser" not in {t.name for t in get_agent("v12_claude").tools}
    assert "browser" in {t.name for t in get_agent("v11_claude").tools}


_V12_V11_PAIRS = [
    ("v12_claude", "v11_claude"),
    ("v12_glm5", "v11_glm5"),
    ("v12_kimi", "v11_kimi"),
]


@pytest.mark.parametrize(("v12", "v11"), _V12_V11_PAIRS)
def test_each_v12_is_its_v11_without_browser_plus_mcp_tools(v12, v11):
    try:
        configure_mcp_tools([_fake_mcp_tool])
        v12_names = {t.name for t in get_agent(v12).tools}
        v11_names = {t.name for t in get_agent(v11).tools}
        assert v12_names == (v11_names - {"browser"}) | {
            "search_mcp_servers",
            "install_mcp_server",
            "list_mcp_servers",
            "remove_mcp_server",
            "_fake_mcp_tool",
        }
    finally:
        configure_mcp_tools([])


@pytest.mark.parametrize(
    ("v12", "v11_tools"),
    [
        ("v12_claude", _V11_CLAUDE_TOOLS),
        ("v12_glm5", _V11_GLM5_TOOLS),
        ("v12_kimi", _V11_KIMI_TOOLS),
    ],
)
def test_each_v12_delegates_to_its_own_v11_subagent(v12, v11_tools):
    v11_subagent = next(t for t in v11_tools if t.name == "run_subagent")
    assert v11_subagent in get_agent(v12).tools


@pytest.mark.parametrize(("v12", "v11"), _V12_V11_PAIRS)
def test_each_v12_uses_its_v11_model(v12, v11):
    v12_agent, v11_agent = get_agent(v12), get_agent(v11)
    assert isinstance(v12_agent, AugAgent)
    assert v12_agent._model_name == v11_agent._model_name
