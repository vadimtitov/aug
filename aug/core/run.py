"""Agent run registry — one AgentRun per in-flight agent thread."""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

# Preprocessed message content ready for the LLM:
# either a plain string or a list of OpenAI-style content blocks
# ({"type": "text", ...}, {"type": "image_url", ...}, etc.)
MessageContent = str | list[dict[str, Any]]

# Key used to store the AgentRun in the LangGraph config's configurable dict,
# allowing tools to access the run's control plane.
AGENT_RUN_CONFIG_KEY = "agent_run"


@dataclass
class AgentRun:
    """Control plane for a single in-flight agent run.

    id                       — unique ID for this run (for logging/correlation).
    user_requested_stop      — set to hard-stop the run; checked by the streaming
                               loop and by tools between execution steps.
    pending_agent_injection  — content queued here is injected as a new HumanMessage
                               at the next interrupt_after=[call_tools] pause point.
    active                   — False once the run has cleaned up; guards against
                               stale injections.
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    user_requested_stop: asyncio.Event = field(default_factory=asyncio.Event)
    pending_agent_injection: asyncio.Queue[MessageContent] = field(default_factory=asyncio.Queue)
    active: bool = True

    def inject_message(self, content: MessageContent) -> None:
        """Queue user content for injection at the next agent pause point."""
        self.pending_agent_injection.put_nowait(content)

    def request_stop(self) -> None:
        """Signal the run to stop at the earliest opportunity."""
        self.user_requested_stop.set()


class RunRegistry:
    """Registry of active AgentRuns, keyed by context_id.

    Use the module-level ``run_registry`` singleton — do not instantiate directly.
    The class boundary makes test isolation (clear()) and future backend swaps
    (e.g. Redis for multi-process deployments) straightforward.
    """

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._run_locks: dict[str, asyncio.Lock] = {}

    def get(self, context_id: str) -> AgentRun | None:
        return self._runs.get(context_id)

    def set(self, context_id: str, run: AgentRun) -> None:
        self._runs[context_id] = run

    def pop(self, context_id: str) -> None:
        self._runs.pop(context_id, None)

    def routing_lock(self, context_id: str) -> asyncio.Lock:
        """Short per-thread lock that serialises routing: inject into the turn in flight,
        or queue a new one.  Never held across a turn."""
        if context_id not in self._locks:
            self._locks[context_id] = asyncio.Lock()
        return self._locks[context_id]

    def run_lock(self, context_id: str) -> asyncio.Lock:
        """The run lock: held for a whole turn by whoever runs one — interfaces, approval
        resumes, pushes, /compact — so two turns never share a context_id at once."""
        if context_id not in self._run_locks:
            self._run_locks[context_id] = asyncio.Lock()
        return self._run_locks[context_id]

    def accepting(self, context_id: str) -> bool:
        """True while a turn is in flight for *context_id* and can still take an
        injected message (it has not been asked to stop)."""
        run = self._runs.get(context_id)
        return run is not None and run.active and not run.user_requested_stop.is_set()

    def clear(self) -> None:
        self._runs.clear()
        self._locks.clear()
        self._run_locks.clear()


run_registry = RunRegistry()
