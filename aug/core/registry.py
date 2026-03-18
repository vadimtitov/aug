"""Agent registry.

To add a new agent, instantiate a BaseAgent subclass and add it to _REGISTRY.
"""

from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent, TimeAwareChatAgent
from aug.core.agents.fake_agent import FakeAgent
from aug.core.prompts import LEGACY_SYSTEM_PROMPT
from aug.core.tools.brave_search import brave_search
from aug.core.tools.browser import browser
from aug.core.tools.fetch_page import fetch_page
from aug.core.tools.gmail import gmail_draft, gmail_read_thread, gmail_search, gmail_send
from aug.core.tools.image_gen import generate_image
from aug.core.tools.memory import forget, recall, remember, update_memory
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
from aug.core.tools.set_reminder import set_reminder

_REGISTRY: dict[str, BaseAgent] = {
    "fake": FakeAgent(),
    "default": TimeAwareChatAgent(
        model="gpt-4o",
        system_prompt=LEGACY_SYSTEM_PROMPT,
        tools=[brave_search, fetch_page, run_bash, remember, recall, update_memory, forget],
        temperature=1.0,
    ),
    "v1_claude": TimeAwareChatAgent(
        model="claude-sonnet-4-6",
        system_prompt=LEGACY_SYSTEM_PROMPT,
        tools=[brave_search, fetch_page, run_bash, remember, recall, update_memory, forget],
        temperature=1.0,
    ),
    "v2_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=[brave_search, fetch_page, run_bash, note],
        temperature=1.0,
    ),
    "v3_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v3_gpt4o": AugAgent(
        model="gpt-4o",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v3_gpt41": AugAgent(
        model="gpt-4.1",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v3_gpt51": AugAgent(
        model="gpt-5.1",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v3_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v3_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=[brave_search, fetch_page, run_bash, note, browser],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_gemini_pro": AugAgent(
        model="gemini-2.5-pro",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_gpt4o": AugAgent(
        model="gpt-4o",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_gpt41": AugAgent(
        model="gpt-4.1",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v4_gpt51": AugAgent(
        model="gpt-5.1",
        tools=[
            brave_search,
            fetch_page,
            run_bash,
            note,
            browser,
            gmail_search,
            gmail_read_thread,
            gmail_send,
            gmail_draft,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v6_gemini_flash": AugAgent(
        model="gemini-2.5-flash",
        tools=[
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
            portainer_list_containers,
            portainer_container_logs,
            portainer_list_stacks,
            portainer_deploy_stack,
            set_reminder,
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
    "v5_claude": AugAgent(
        model="claude-sonnet-4-6",
        tools=[
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
        ],
        temperature=1.0,
        recursion_limit=100,
    ),
}

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
    portainer_list_containers,
    portainer_container_logs,
    portainer_container_action,
    portainer_list_stacks,
    portainer_deploy_stack,
    portainer_stack_action,
    set_reminder,
]

_REGISTRY.update(
    {
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
    }
)


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
