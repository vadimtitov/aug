"""The "scripted" agent: a real AugAgent whose model is the fake LLM in scripts/fakes.py.

Registered only when ``SCRIPTED_AGENT`` is set, which only the scenario stack
(docker-compose.scenarios.yml) does.  That stack also points ``LLM_BASE_URL`` at the
fake, so every model call — the agent's, compaction's, consolidation's — is scripted
and free.  The tools below exist so scenarios can exercise progress, approvals and
/stop without touching anything real.
"""

import asyncio

from langchain_core.tools import tool

from aug.core.agents.chat_agent import AugAgent
from aug.core.harness.types import ToolProgress
from aug.core.tools.approval import requires_approval
from aug.core.tools.context import current_tool_context
from aug.core.tools.note import note
from aug.core.tools.respond_with_file import respond_with_file
from aug.core.tools.skills import save_skill

SCRIPTED_MODEL = "scripted"


@tool
async def dev_progress(steps: int) -> str:
    """Report *steps* progress updates, then finish."""
    ctx = current_tool_context()
    for i in range(1, steps + 1):
        if ctx is not None:
            await ctx.emit(ToolProgress(text=f"Step {i} of {steps}"))
        await asyncio.sleep(0.05)
    return f"Completed {steps} steps."


@tool
@requires_approval(describe=lambda command: ("dev", command))
async def dev_guarded(command: str) -> str:
    """Run *command*, which needs the user's approval first."""
    return f"Ran: {command}"


@tool
async def dev_wait(seconds: float) -> str:
    """Wait *seconds*, stopping early if the user asks the run to stop."""
    ctx = current_tool_context()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        if ctx is not None and ctx.stopped.is_set():
            return "Stopped by the user before the wait finished."
        await asyncio.sleep(0.05)
    return f"Waited {seconds} seconds."


def build_scripted_agent() -> AugAgent:
    return AugAgent(
        model=SCRIPTED_MODEL,
        tools=[dev_progress, dev_guarded, dev_wait, respond_with_file, note, save_skill],
        temperature=0.0,
        recursion_limit=100,
        compaction_model=SCRIPTED_MODEL,
        compaction_threshold=0.8,
        context_window=200_000,
        max_summary_tokens=500,
    )
