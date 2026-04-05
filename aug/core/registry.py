"""Agent registry.

To add a new agent, instantiate a BaseAgent subclass and add it to _REGISTRY.
"""

from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent
from aug.core.agents.fake_agent import FakeAgent
from aug.core.reflexes.homeassistant import homeassistant_reflex
from aug.core.tools.brave_search import brave_search
from aug.core.tools.browser import browser
from aug.core.tools.fetch_page import fetch_page
from aug.core.tools.gmail import gmail_draft, gmail_read_thread, gmail_search, gmail_send
from aug.core.tools.image_gen import edit_image, generate_image
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
from aug.core.tools.set_reminder import set_reminder
from aug.core.tools.skills import delete_skill, get_skill, save_skill, write_skill_file

_SKILLS_TOOLS = [get_skill, save_skill, write_skill_file, delete_skill]

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
    set_reminder,
    *_SKILLS_TOOLS,
]

_V9_TOOLS = [
    *_V7_TOOLS,
    run_ssh,
    list_ssh_targets,
    download_ssh_file,
    upload_ssh_file,
]

_V8_REFLEXES = [homeassistant_reflex("gemini-2.5-flash-lite")]

_REGISTRY: dict[str, BaseAgent] = {
    "fake": FakeAgent(),
    "v7_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gpt4o": AugAgent(
        model="gpt-4o",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gpt51": AugAgent(
        model="gpt-5.1",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gpt53": AugAgent(
        model="gpt-5.3-chat-latest",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v7_gpt54": AugAgent(
        model="gpt-5.4",
        tools=_V7_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_gpt4o": AugAgent(
        model="gpt-4o",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_gpt51": AugAgent(
        model="gpt-5.1",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v8_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V7_TOOLS,
        reflexes=_V8_REFLEXES,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_gpt4o": AugAgent(
        model="gpt-4o",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_gpt41": AugAgent(
        model="gpt-4.1",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_gpt51": AugAgent(
        model="gpt-5.1",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
    ),
    "v9_glm5": AugAgent(
        model="glm-5",
        tools=_V9_TOOLS,
        temperature=1.0,
        recursion_limit=100,
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
