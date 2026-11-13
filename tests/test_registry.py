"""The registry: the v13_kimi profiles, their tools, tiers, and the move off deleted versions."""

import inspect
from unittest.mock import patch

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from aug.core.agents.chat_agent import AugAgent
from aug.core.harness.types import ModelTier
from aug.core.registry import (
    base_tool_names,
    configure_harnesses,
    configure_mcp_tools,
    get_agent,
    get_harness,
    list_agents,
    migrate_conversations,
)
from aug.utils.file_settings import AppSettings, ConversationSettings

PROFILES = ["v13_kimi", "v13_glm"]


@tool
def _fake_mcp_tool() -> str:
    """A fake MCP-sourced tool for tests."""
    return "ok"


@pytest.fixture(autouse=True)
def _no_mcp_tools():
    configure_mcp_tools([])
    yield
    configure_mcp_tools([])


def test_only_the_current_profiles_are_registered():
    assert list_agents() == ["fake", *PROFILES]


def test_profile_models():
    assert get_agent("v13_kimi")._model_name == "accounts/fireworks/models/kimi-k3"
    assert get_agent("v13_glm")._model_name == "accounts/fireworks/models/glm-5p3"


@pytest.mark.parametrize("name", PROFILES)
def test_profiles_browse_through_mcp_not_the_browser_tool(name):
    names = _tool_names(name)
    assert "browser" not in names
    assert {"respond_with_file", "run_subagent", "search_mcp_servers", "install_mcp_server"} <= (
        names
    )


def test_describe_image_is_only_on_the_profile_without_vision():
    assert "describe_image" in _tool_names("v13_glm")
    assert "describe_image" not in _tool_names("v13_kimi")


def test_both_profiles_share_one_tool_list_apart_from_describe_image():
    assert _tool_names("v13_kimi") == _tool_names("v13_glm") - {"describe_image"}


@pytest.mark.parametrize("name", PROFILES)
def test_mcp_tools_reach_the_profile_and_its_subagent(name):
    configure_mcp_tools([_fake_mcp_tool])
    agent = get_agent(name)

    assert "_fake_mcp_tool" in _tool_names(name)
    subagent = _subagent_of(agent)
    assert "_fake_mcp_tool" in {t.name for t in subagent.tools}
    assert "run_subagent" not in {t.name for t in subagent.tools}
    assert subagent._model_name == agent._model_name


@pytest.mark.parametrize("name", PROFILES)
def test_reconfiguring_replaces_the_mcp_tools(name):
    configure_mcp_tools([_fake_mcp_tool])
    configure_mcp_tools([])

    assert "_fake_mcp_tool" not in _tool_names(name)
    assert "_fake_mcp_tool" not in {t.name for t in _subagent_of(get_agent(name)).tools}


def test_base_tool_names_cover_every_native_tool_and_no_mcp_one():
    configure_mcp_tools([_fake_mcp_tool])
    names = base_tool_names()

    for profile in PROFILES:
        assert _tool_names(profile) - {"_fake_mcp_tool"} <= names
    assert "_fake_mcp_tool" not in names


def test_scenario_tools_are_added_only_on_the_scenario_stack():
    with patch("aug.core.registry.get_settings") as settings:
        settings.return_value.SCRIPTED_AGENT = True
        configure_mcp_tools([])
        assert "dev_guarded" in _tool_names("v13_kimi")
        assert "scripted" in list_agents()
    configure_mcp_tools([])
    assert "dev_guarded" not in _tool_names("v13_kimi")


# ---------------------------------------------------------------------------
# oneshot() tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "strong"),
    [
        ("v13_kimi", "accounts/fireworks/models/kimi-k3"),
        ("v13_glm", "accounts/fireworks/models/glm-5p3"),
    ],
)
def test_oneshot_tiers(name, strong):
    configure_harnesses(MemorySaver())
    harness = get_harness(name)

    assert harness.model_for(ModelTier.STRONG) == strong
    assert harness.model_for(ModelTier.FAST) == "accounts/fireworks/models/glm-5p3-flash"


# ---------------------------------------------------------------------------
# Chats on deleted versions
# ---------------------------------------------------------------------------


def test_chats_on_deleted_versions_move_to_a_current_profile():
    settings = AppSettings(
        conversations={
            "tg-1": ConversationSettings(agent="v11_kimi"),
            "tg-2": ConversationSettings(agent="v12_glm5_fast"),
            "tg-3": ConversationSettings(agent="v9_claude"),
            "tg-4": ConversationSettings(agent="v13_glm"),
            "tg-5": ConversationSettings(agent="default"),
        }
    )
    saved: list[AppSettings] = []
    with (
        patch("aug.core.registry.load_settings", return_value=settings),
        patch("aug.core.registry.save_settings", side_effect=saved.append),
    ):
        moved = migrate_conversations()

    assert moved == ["tg-1", "tg-2", "tg-3"]
    agents = {k: v.agent for k, v in saved[0].conversations.items()}
    assert agents == {
        "tg-1": "v13_kimi",
        "tg-2": "v13_glm",
        "tg-3": "v13_kimi",
        "tg-4": "v13_glm",
        "tg-5": "default",
    }


def test_nothing_is_saved_when_no_chat_is_on_a_deleted_version():
    settings = AppSettings(conversations={"tg-1": ConversationSettings(agent="v13_kimi")})
    with (
        patch("aug.core.registry.load_settings", return_value=settings),
        patch("aug.core.registry.save_settings") as save,
    ):
        assert migrate_conversations() == []
    save.assert_not_called()


def _tool_names(name: str) -> set[str]:
    return {t.name for t in get_agent(name).tools}


def _subagent_of(agent: AugAgent) -> AugAgent:
    """The subagent behind *agent*'s run_subagent tool."""
    run_subagent = next(t for t in agent.tools if t.name == "run_subagent")
    return inspect.getclosurevars(run_subagent.coroutine).nonlocals["subagent"]
