"""Agent registry.

To add a new agent, instantiate a BaseAgent subclass and add it to _REGISTRY.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel as CompiledGraph

from aug.core.agents.base_agent import BaseAgent
from aug.core.agents.chat_agent import AugAgent, TimeAwareChatAgent
from aug.core.agents.fake_agent import FakeAgent
from aug.core.tools.brave_search import brave_search
from aug.core.tools.browser import browser
from aug.core.tools.fetch_page import fetch_page
from aug.core.tools.memory import forget, recall, remember, update_memory
from aug.core.tools.note import note
from aug.core.tools.run_bash import run_bash

_SYSTEM_PROMPT = (
    "You are AUG. You are a razor-sharp personal assistant — think Jarvis, not a chatbot. "
    "You have a dry wit, speak like a brilliant friend who happens to know everything, "
    "and get straight to the point without padding or filler. "
    "You genuinely try before answering: if there's any chance your knowledge is outdated "
    "or you're not 100% sure, you use your search tool to verify before responding — "
    "never guess, never hallucinate. "
    "When you search, you search properly: read the results, synthesise them, "
    "and give a crisp answer — not a list of links. "
    "You're concise by default but thorough when it matters. "
    "You have opinions, you push back when something doesn't add up, "
    "and you treat the user as an intelligent adult. "
    "When multiple tools are needed, call them simultaneously rather than one at a time."
)

_REGISTRY: dict[str, BaseAgent] = {
    "fake": FakeAgent(),
    "default": TimeAwareChatAgent(
        model="gpt-4o",
        system_prompt=_SYSTEM_PROMPT,
        tools=[brave_search, fetch_page, run_bash, remember, recall, update_memory, forget],
        temperature=1.0,
    ),
    "v1_claude": TimeAwareChatAgent(
        model="claude-sonnet-4-6",
        system_prompt=_SYSTEM_PROMPT,
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
}

# Module-level cache so we don't recompile the same graph on every request.
_graph_cache: dict[str, CompiledGraph] = {}


def list_agents() -> list[str]:
    """Return all registered agent names."""
    return list(_REGISTRY.keys())


def get_agent(name: str, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
    """Return a compiled graph for *name*, building it on first call.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    if name not in _REGISTRY:
        registered = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown agent '{name}'. Registered agents: {registered}")

    if name not in _graph_cache:
        _graph_cache[name] = _REGISTRY[name].build(checkpointer)

    return _graph_cache[name]
