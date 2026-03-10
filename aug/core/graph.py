"""Agent registry and graph factory.

To add a new agent:
1. Create ``aug/core/agents/<name>.py`` with a class that inherits ``BaseAgent``.
2. Add an instance to ``_REGISTRY`` below.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel as CompiledGraph

from aug.core.agents.base import BaseAgent
from aug.core.agents.default import DefaultAgent
from aug.core.agents.fake import FakeAgent

_REGISTRY: dict[str, BaseAgent] = {
    "default": DefaultAgent(),
    "fake": FakeAgent(),
}

# Module-level cache so we don't recompile the same graph on every request.
_graph_cache: dict[str, CompiledGraph] = {}


def list_agents() -> list[str]:
    """Return all registered agent version strings."""
    return list(_REGISTRY.keys())


def get_agent(version: str, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
    """Return a compiled graph for *version*, building it on first call.

    Raises:
        ValueError: if *version* is not in the registry.
    """
    if version not in _REGISTRY:
        registered = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown agent '{version}'. Registered agents: {registered}")

    if version not in _graph_cache:
        _graph_cache[version] = _REGISTRY[version].build(checkpointer)

    return _graph_cache[version]
