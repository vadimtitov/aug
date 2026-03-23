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


class StopSignal:
    """Sentinel placed on pending_tool_instruction by request_stop() to signal tools to abort.

    Using an object instead of a string prevents collision with actual user messages.
    """


TOOL_STOP_SENTINEL = StopSignal()


@dataclass
class AgentRun:
    """Control plane for a single in-flight agent run.

    id                       — unique ID for this run (for logging/correlation).
    user_requested_stop      — set to hard-stop the run; the streaming loop
                               checks this flag between every yielded event.
    pending_agent_injection  — content queued here is injected as a new HumanMessage
                               at the next interrupt_after=[call_tools] pause point.
    pending_tool_instruction — text queued here is readable by tools that support
                               mid-execution steering (e.g. browser), checked
                               between steps via the tool's progress callback.
    active                   — False once the run has cleaned up; guards against
                               stale injections.
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    user_requested_stop: asyncio.Event = field(default_factory=asyncio.Event)
    pending_agent_injection: asyncio.Queue[MessageContent] = field(default_factory=asyncio.Queue)
    pending_tool_instruction: asyncio.Queue[str | StopSignal] = field(default_factory=asyncio.Queue)
    active: bool = True

    def inject_message(self, content: MessageContent) -> None:
        """Forward user content into the active run.

        Always queued for the agent loop (next interrupt pause).
        Only queued for tool steering if content is plain text — multimodal
        content is not actionable as a tool instruction.
        """
        self.pending_agent_injection.put_nowait(content)
        if isinstance(content, str):
            self.pending_tool_instruction.put_nowait(content)

    def request_stop(self) -> None:
        """Signal the run to stop at the earliest opportunity."""
        self.user_requested_stop.set()
        self.pending_tool_instruction.put_nowait(TOOL_STOP_SENTINEL)


class RunRegistry:
    """Registry of active AgentRuns, keyed by thread_id.

    Use the module-level ``run_registry`` singleton — do not instantiate directly.
    The class boundary makes test isolation (clear()) and future backend swaps
    (e.g. Redis for multi-process deployments) straightforward.
    """

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, thread_id: str) -> AgentRun | None:
        return self._runs.get(thread_id)

    def set(self, thread_id: str, run: AgentRun) -> None:
        self._runs[thread_id] = run

    def pop(self, thread_id: str) -> None:
        self._runs.pop(thread_id, None)

    def thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Per-thread lock to serialise routing decisions in BaseInterface.run()."""
        if thread_id not in self._locks:
            self._locks[thread_id] = asyncio.Lock()
        return self._locks[thread_id]

    def clear(self) -> None:
        self._runs.clear()
        self._locks.clear()


run_registry = RunRegistry()
