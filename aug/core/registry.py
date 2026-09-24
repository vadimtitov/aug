"""Agent registry.

To add a new agent, instantiate a BaseAgent subclass and add it to _REGISTRY.
"""

from langchain_core.tools import BaseTool

from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent
from aug.core.agents.fake_agent import FakeAgent
from aug.core.reflexes.homeassistant import homeassistant_reflex
from aug.core.tools.brave_search import brave_search
from aug.core.tools.browser import browser
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

_V7_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    # set_reminder disabled — use create_task with push_type="forward"/"inject" instead
]

_V9_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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

_V8_REFLEXES = [homeassistant_reflex("gemini-2.5-flash-lite")]

# Subagent tools: same capabilities as the main agent but without run_subagent (no nesting).
_SUBAGENT_TOOLS = [
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

# Per-model subagents — each V11 agent gets a subagent using the same model.
_subagent_claude = AugAgent(
    model="claude-sonnet-4-6",
    tools=_SUBAGENT_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    compaction_model="claude-haiku-4-5",
    compaction_threshold=0.7,
    context_window=500_000,
    max_summary_tokens=2000,
)
_subagent_gpt41 = AugAgent(
    model="gpt-4.1",
    tools=_SUBAGENT_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    compaction_model="gpt-4.1",
    compaction_threshold=0.7,
    context_window=1_000_000,
    max_summary_tokens=2000,
)
_subagent_gemini_pro = AugAgent(
    model="gemini-2.5-pro",
    tools=_SUBAGENT_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    compaction_model="gemini-2.5-pro",
    compaction_threshold=0.7,
    context_window=1_000_000,
    max_summary_tokens=2000,
)
_subagent_glm5 = AugAgent(
    model="accounts/fireworks/models/glm-5p2",
    tools=_SUBAGENT_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    vision_description_model="gemini-2.5-flash",
    compaction_model="accounts/fireworks/models/glm-5p2",
    compaction_threshold=0.8,
    context_window=200_000,
    max_summary_tokens=2000,
)
_subagent_kimi = AugAgent(
    model="accounts/fireworks/models/kimi-k2p6",
    tools=_SUBAGENT_TOOLS,
    temperature=0.0,
    recursion_limit=50,
    compaction_model="accounts/fireworks/models/kimi-k2p6",
    compaction_threshold=0.8,
    context_window=200_000,
    max_summary_tokens=2000,
)

_V11_CLAUDE_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    make_run_subagent_tool(_subagent_claude),
]

_V11_GPT41_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    make_run_subagent_tool(_subagent_gpt41),
]

_V11_GEMINI_PRO_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    make_run_subagent_tool(_subagent_gemini_pro),
]

_V11_GLM5_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    make_run_subagent_tool(_subagent_glm5),
]

_V11_KIMI_TOOLS = [
    brave_search,
    fetch_page,
    run_bash,
    note,
    browser,
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
    make_run_subagent_tool(_subagent_kimi),
]

# v12_claude = v11_claude's tools + MCP management tools + whatever MCP servers
# are actually reachable at startup. V11 agents are immutable and untouched —
# this is a new version, not a modification of _V11_CLAUDE_TOOLS.
_V12_BASE_TOOLS = [
    *_V11_CLAUDE_TOOLS,
    search_mcp_servers,
    install_mcp_server,
    list_mcp_servers,
    remove_mcp_server,
]


def v12_base_tool_names() -> set[str]:
    """Names of v12_claude's non-MCP tools — seeds MCPManager's collision check
    so a server can never shadow a native AUG tool."""
    return {t.name for t in _V12_BASE_TOOLS}


def configure_mcp_tools(tools: list[BaseTool]) -> None:
    """Rebuild v12_claude with the MCP tools MCPManager actually connected.

    Called once from aug/app.py's lifespan(), after MCPManager.load_all()
    resolves which configured servers are reachable. Before this runs (e.g. in
    tests, or if a server config exists but the app hasn't booted through
    lifespan) v12_claude simply has no MCP-sourced tools yet — same graceful
    degradation as "every MCP server failed to connect".
    """
    _REGISTRY["v12_claude"] = _build_v12_claude(tools)


_REGISTRY: dict[str, BaseAgent] = {
    "fake": FakeAgent(),
    "subagent": _subagent_claude,
    "v9_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_gpt4o": AugAgent(
        model="gpt-4o",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_gpt51": AugAgent(
        model="gpt-5.1",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
    ),
    "v9_glm5": AugAgent(
        model="accounts/fireworks/models/glm-5p2",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        vision_description_model="gemini-2.5-flash",
    ),
    "v10_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="claude-haiku-4-5",
        compaction_threshold=0.7,
        context_window=500_000,
        max_summary_tokens=2000,
    ),
    "v10_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="gpt-4.1",
        compaction_threshold=0.7,
        context_window=1_000_000,
        max_summary_tokens=2000,
    ),
    "v10_gpt51": AugAgent(
        model="gpt-5.1",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="gpt-5.1",
        compaction_threshold=0.7,
        context_window=200_000,
        max_summary_tokens=2000,
    ),
    "v10_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="gemini-2.5-pro",
        compaction_threshold=0.7,
        context_window=1_000_000,
        max_summary_tokens=2000,
    ),
    "v10_glm5": AugAgent(
        model="accounts/fireworks/models/glm-5p2",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        vision_description_model="gemini-2.5-flash",
        compaction_model="accounts/fireworks/models/glm-5p2",
        compaction_threshold=0.8,
        context_window=200_000,
        max_summary_tokens=2000,
    ),
    "v10_kimi": AugAgent(
        model="accounts/fireworks/models/kimi-k2p6",
        tools=_V9_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="accounts/fireworks/models/kimi-k2p6",
        compaction_threshold=0.8,
        context_window=200_000,
        max_summary_tokens=2000,
    ),
    "v11_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V11_CLAUDE_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="claude-haiku-4-5",
        compaction_threshold=0.7,
        context_window=500_000,
        max_summary_tokens=2000,
    ),
    "v11_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V11_GPT41_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="gpt-4.1",
        compaction_threshold=0.7,
        context_window=1_000_000,
        max_summary_tokens=2000,
    ),
    "v11_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V11_GEMINI_PRO_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="gemini-2.5-pro",
        compaction_threshold=0.7,
        context_window=1_000_000,
        max_summary_tokens=2000,
    ),
    "v11_glm5": AugAgent(
        model="accounts/fireworks/models/glm-5p2",
        tools=_V11_GLM5_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        vision_description_model="gemini-2.5-flash",
        compaction_model="accounts/fireworks/models/glm-5p2",
        compaction_threshold=0.8,
        context_window=200_000,
        max_summary_tokens=2000,
    ),
    "v11_kimi": AugAgent(
        model="accounts/fireworks/models/kimi-k2p6",
        tools=_V11_KIMI_TOOLS,
        temperature=0.0,
        recursion_limit=100,
        compaction_model="accounts/fireworks/models/kimi-k2p6",
        compaction_threshold=0.8,
        context_window=200_000,
        max_summary_tokens=2000,
    ),
}


def list_agents() -> list[str]:
    """Return all registered agent names."""
    return list(_REGISTRY.keys())


def get_agent(name: str) -> BaseAgent:
    """Return the agent for *name*.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    if name not in _REGISTRY:
        registered = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown agent '{name}'. Registered agents: {registered}")
    return _REGISTRY[name]


def _build_v12_claude(mcp_tools: list[BaseTool]) -> AugAgent:
    return AugAgent(
        model="claude-sonnet-4-6",
        tools=[*_V12_BASE_TOOLS, *mcp_tools],
        temperature=0.0,
        recursion_limit=100,
        compaction_model="claude-haiku-4-5",
        compaction_threshold=0.7,
        context_window=500_000,
        max_summary_tokens=2000,
    )


# Seeds v12_claude with no MCP tools until aug/app.py's lifespan() calls
# configure_mcp_tools() again with whatever MCPManager actually connected.
configure_mcp_tools([])
