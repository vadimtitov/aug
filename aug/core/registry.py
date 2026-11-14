"""Agent registry: one entry per profile (agent version).

``v13_kimi`` (kimi-k3) and ``v13_glm`` (glm-5.3) run on LangGraph; ``v14_claude_code``
(Claude Opus 5.5) runs on Claude Code.  All are built by ``_build_profiles()``, since
their tool lists include whichever MCP servers connected at startup.  The ``browser``
tool's code stays but no profile exposes it.
"""

import logging

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from aug.config import get_settings
from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent
from aug.core.agents.fake_agent import FakeAgent
from aug.core.harness.base import Harness
from aug.core.harness.claude_code.harness import ClaudeCodeHarness, ClaudeCodeProfile
from aug.core.harness.langgraph.harness import LangGraphHarness
from aug.core.harness.types import ModelTier
from aug.core.tools.brave_search import brave_search
from aug.core.tools.fetch_page import fetch_page
from aug.core.tools.gmail import gmail_draft, gmail_read_thread, gmail_search, gmail_send
from aug.core.tools.image_gen import edit_image, generate_image
from aug.core.tools.mcp import (
    install_mcp_server,
    list_mcp_servers,
    remove_mcp_server,
    search_mcp_servers,
)
from aug.core.tools.note import note
from aug.core.tools.portainer import (
    portainer_container_action,
    portainer_container_logs,
    portainer_deploy_stack,
    portainer_list_containers,
    portainer_list_stacks,
    portainer_stack_action,
)
from aug.core.tools.respond_with_file import respond_with_file
from aug.core.tools.run_bash import run_bash
from aug.core.tools.run_ssh import download_ssh_file, list_ssh_targets, run_ssh, upload_ssh_file
from aug.core.tools.skills import delete_skill, get_skill, save_skill, write_skill_file
from aug.core.tools.subagent import make_run_subagent_tool
from aug.core.tools.tasks import create_task, delete_task, list_tasks, update_task
from aug.dev.scripted import SCRIPTED_TOOLS, build_scripted_agent
from aug.utils.file_settings import load_settings, save_settings

logger = logging.getLogger(__name__)

_KIMI = "accounts/fireworks/models/kimi-k3"
_GLM = "accounts/fireworks/models/glm-5p3"
# oneshot()'s FAST tier on both profiles: what jobs (the dream) run on.
_FAST = "accounts/fireworks/models/glm-5p3-flash"
# Describes images for v13_glm, whose model has no vision (describe_image).
_VISION_DESCRIPTION_MODEL = _KIMI

_CLAUDE_STRONG = "claude-opus-5-5"
_CLAUDE_FAST = "claude-haiku-5-5"

# Profiles a chat on a deleted agent version is moved to: glm versions to v13_glm,
# every other one to v13_kimi.
DEFAULT_PROFILE = "v13_kimi"
_GLM_PROFILE = "v13_glm"

_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    gmail_search,
    gmail_read_thread,
    gmail_send,
    gmail_draft,
    respond_with_file,
    generate_image,
    edit_image,
    portainer_list_containers,
    portainer_container_logs,
    portainer_container_action,
    portainer_list_stacks,
    portainer_deploy_stack,
    portainer_stack_action,
    get_skill,
    save_skill,
    write_skill_file,
    delete_skill,
    create_task,
    list_tasks,
    update_task,
    delete_task,
    run_ssh,
    list_ssh_targets,
    download_ssh_file,
    upload_ssh_file,
]
# Claude Code's profile: AUG's tools except those Claude has its own of (run_bash: Bash,
# fetch_page: WebFetch, brave_search: WebSearch, get_skill: Skill).
_CLAUDE_TOOLS = [t for t in _TOOLS if t not in (run_bash, fetch_page, brave_search, get_skill)]
# AUG's tools Claude keeps loaded; the rest it looks up through ToolSearch when needed.
_CLAUDE_ALWAYS_LOADED = frozenset({"respond_with_file", "note", "create_task", "list_tasks"})
# Claude's own tools a profile may use.  Anything else, including a tool a new CLI
# version adds, stays blocked: Claude's cron and wake-ups (scheduling is AUG's), remote
# triggers, peer messaging (it could carry text between chats), worktrees.
_CLAUDE_BUILTIN_TOOLS = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "WebSearch",
    "WebFetch",
    "Task",
    "TaskStop",
    "Monitor",
    "Skill",
    "ToolSearch",
)

_MCP_MANAGEMENT_TOOLS = [
    search_mcp_servers,
    install_mcp_server,
    list_mcp_servers,
    remove_mcp_server,
]

# A profile's subagent: the same model and tools (plus the MCP servers'), without
# run_subagent, so it cannot nest.
_subagent_kimi = AugAgent(
    model=_KIMI,
    tools=_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    compaction_model=_KIMI,
    compaction_threshold=0.8,
    context_window=200_000,
    max_summary_tokens=2000,
)
_subagent_glm = AugAgent(
    model=_GLM,
    tools=_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    vision_description_model=_VISION_DESCRIPTION_MODEL,
    compaction_model=_GLM,
    compaction_threshold=0.8,
    context_window=200_000,
    max_summary_tokens=2000,
)

# oneshot()'s model per tier, per profile.  A profile not listed calls its own model.
_TIER_MODELS: dict[str, dict[ModelTier, str]] = {
    "v13_kimi": {ModelTier.STRONG: _KIMI, ModelTier.FAST: _FAST},
    "v13_glm": {ModelTier.STRONG: _GLM, ModelTier.FAST: _FAST},
}


def base_tool_names() -> set[str]:
    """Names of the profiles' non-MCP tools: seeds MCPManager's collision check so a
    server can never shadow a native AUG tool."""
    return {t.name for t in [*_TOOLS, *_MCP_MANAGEMENT_TOOLS, *SCRIPTED_TOOLS]} | {
        "run_subagent",
        "describe_image",
    }


def configure_mcp_tools(tools: list[BaseTool]) -> None:
    """Rebuild the profiles with the MCP tools MCPManager actually connected.

    Called once from aug/app.py's lifespan(), after MCPManager.load_all() resolves which
    configured servers are reachable.  Until then (in tests, say) the profiles have no
    MCP tools, the same as every server failing to connect.
    """
    _REGISTRY.update(_build_profiles(tools))
    if get_settings().SCRIPTED_AGENT:
        _REGISTRY["scripted"] = build_scripted_agent()
    else:
        _REGISTRY.pop("scripted", None)


def migrate_conversations() -> list[str]:
    """Move every conversation set to an agent version that no longer exists onto a
    current profile (glm versions to v13_glm, the rest to v13_kimi).  Returns the moved
    conversation ids.  Called once at startup, before any message is taken."""
    settings = load_settings()
    moved = []
    for conversation_id, conversation in settings.conversations.items():
        if conversation.agent in _REGISTRY or conversation.agent == "default":
            continue
        target = _GLM_PROFILE if "glm" in conversation.agent else DEFAULT_PROFILE
        logger.info(
            "agent version %s is gone: %s moves to %s", conversation.agent, conversation_id, target
        )
        conversation.agent = target
        moved.append(conversation_id)
    if moved:
        save_settings(settings)
    return moved


# A profile: a LangGraph agent, or a Claude Code profile.
Profile = BaseAgent | ClaudeCodeProfile

_REGISTRY: dict[str, Profile] = {
    "fake": FakeAgent(),
}


def list_agents() -> list[str]:
    """Return all registered agent names."""
    return list(_REGISTRY.keys())


def get_agent(name: str) -> Profile:
    """Return the profile for *name*.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    if name not in _REGISTRY:
        registered = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown agent '{name}'. Registered agents: {registered}")
    return _REGISTRY[name]


def configure_harnesses(checkpointer: BaseCheckpointSaver) -> None:
    """Give every agent version's harness the app's checkpointer.  Called once from
    aug/app.py's lifespan(), before any turn runs."""
    _harnesses.configure(checkpointer)


def get_harness(name: str) -> Harness:
    """Return the harness that runs agent version *name*.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    return _harnesses.get(name)


def harnesses() -> list[Harness]:
    """Every agent version's harness, for the lifespan to enter."""
    return [_harnesses.get(name) for name in _REGISTRY]


class _Harnesses:
    """One harness per agent version, built on first use.

    Keyed by the agent object too: configure_mcp_tools() rebuilds the profiles, and a
    LangGraph harness must never keep running the version it replaced.  A Claude Code
    harness is kept and given the new profile instead, so its warm processes are never
    orphaned.
    """

    def __init__(self) -> None:
        self._checkpointer: BaseCheckpointSaver | None = None
        self._built: dict[str, tuple[Profile, Harness]] = {}

    def configure(self, checkpointer: BaseCheckpointSaver) -> None:
        self._checkpointer = checkpointer
        self._built.clear()

    def get(self, name: str) -> Harness:
        agent = get_agent(name)
        built = self._built.get(name)
        if built is not None and built[0] is agent:
            return built[1]
        harness: Harness
        if isinstance(agent, ClaudeCodeProfile) and built is not None:
            # The same harness takes the new profile: it was entered into running()
            # and holds warm processes, which restart on their next turn if the
            # tools changed.
            harness = built[1]
            assert isinstance(harness, ClaudeCodeHarness)
            harness.profile = agent
        elif isinstance(agent, ClaudeCodeProfile):
            harness = ClaudeCodeHarness(agent)
        elif self._checkpointer is None:
            raise RuntimeError("configure_harnesses() has not run yet")
        else:
            harness = LangGraphHarness(agent, self._checkpointer, models=_TIER_MODELS.get(name))
        self._built[name] = (agent, harness)
        return harness


_harnesses = _Harnesses()


def _build_profiles(mcp_tools: list[BaseTool]) -> dict[str, Profile]:
    # Dev only: the scenario stack runs its scripts on the real profiles too.
    dev_tools = SCRIPTED_TOOLS if get_settings().SCRIPTED_AGENT else []
    return {
        "v13_kimi": AugAgent(
            model=_KIMI,
            tools=[
                *_TOOLS,
                *_MCP_MANAGEMENT_TOOLS,
                make_run_subagent_tool(_subagent_kimi.with_extra_tools(mcp_tools)),
                *mcp_tools,
                *dev_tools,
            ],
            temperature=0.0,
            recursion_limit=100,
            compaction_model=_KIMI,
            compaction_threshold=0.8,
            context_window=200_000,
            max_summary_tokens=2000,
        ),
        "v13_glm": AugAgent(
            model=_GLM,
            tools=[
                *_TOOLS,
                *_MCP_MANAGEMENT_TOOLS,
                make_run_subagent_tool(_subagent_glm.with_extra_tools(mcp_tools)),
                *mcp_tools,
                *dev_tools,
            ],
            temperature=0.0,
            recursion_limit=100,
            vision_description_model=_VISION_DESCRIPTION_MODEL,
            compaction_model=_GLM,
            compaction_threshold=0.8,
            context_window=200_000,
            max_summary_tokens=2000,
        ),
        "v14_claude_code": ClaudeCodeProfile(
            model=_CLAUDE_STRONG,
            effort="high",
            builtin_tools=_CLAUDE_BUILTIN_TOOLS,
            tools=[*_CLAUDE_TOOLS, *_MCP_MANAGEMENT_TOOLS, *mcp_tools, *dev_tools],
            always_loaded=_CLAUDE_ALWAYS_LOADED,
            models={ModelTier.STRONG: _CLAUDE_STRONG, ModelTier.FAST: _CLAUDE_FAST},
        ),
    }


# Seeds the profiles with no MCP tools until aug/app.py's lifespan() calls
# configure_mcp_tools() again with whatever MCPManager actually connected.
configure_mcp_tools([])
