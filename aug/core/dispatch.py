"""Proactive delivery engine for scheduled tasks and external pushes.

Two public entry points:

fire_task(task_id)
    Called by APScheduler when a scheduled task fires.  Loads the task from
    the database, then delegates to fire_push.

fire_push(app, *, interface, thread_id, message, push_type, isolated, ...)
    Resolves the target thread via the interface, then either:
      - delivers the message directly (push_type="forward"), or
      - runs a constrained agent turn and delivers the final response
        (push_type="agent").

External agent runs are capped at _PUSH_RECURSION_LIMIT iterations.
Tool restriction (e.g. excluding run_bash) is a TODO: it requires either a
dedicated agent variant or a hook in the agent graph to filter tool calls.
"""

import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from aug.core.agents.base_agent import BaseAgent
from aug.core.events import AgentEvent
from aug.core.registry import get_agent
from aug.core.state import AgentState
from aug.utils.db import get_pool
from aug.utils.file_settings import TelegramChatSettings, load_settings
from aug.utils.tasks import get_task

InterfaceName = Literal["telegram", "rest_api"]

logger = logging.getLogger(__name__)

# Maximum agentic loop iterations allowed for externally-triggered agent runs.
_PUSH_RECURSION_LIMIT = 10

# Retry configuration for fire_task delivery failures.
_MAX_TASK_RETRIES = 10
_RETRY_DELAY_CAP_MINUTES = 60
# Prefix used for APScheduler retry job IDs — must not collide with task UUIDs.
TASK_RETRY_JOB_PREFIX = "retry-"

# Regex to extract the chat_id portion from a tg-{chat_id}-… thread ID.
_TG_CHAT_ID_RE = re.compile(r"^tg-(-?\d+)-")

# FastAPI app reference stored at startup so APScheduler job functions can
# reach app state without needing it as a serialisable argument.
_app: FastAPI | None = None


def set_app(app: FastAPI) -> None:
    """Store the FastAPI app for use by APScheduler job functions."""
    global _app
    _app = app


async def fire_task(task_id: str, retry_count: int = 0) -> None:
    """Entry point called by APScheduler when a scheduled task fires.

    Loads the task from the database, skips if disabled, then delegates to
    fire_push.  On delivery failure, schedules an exponential-backoff retry job
    (up to _MAX_TASK_RETRIES attempts) before dead-lettering.
    """
    if _app is None:
        logger.error("fire_task: app not registered (task_id=%s)", task_id)
        return

    async with get_pool().acquire() as conn:
        task = await get_task(conn, task_id)

    if task is None:
        logger.warning("fire_task: task not found task_id=%s", task_id)
        return
    if not task.enabled:
        return
    if task.interface not in {"telegram", "rest_api"}:
        logger.error("fire_task: unknown interface %r task_id=%s", task.interface, task_id)
        return

    logger.info("fire_task: firing task_id=%s name=%s attempt=%d", task_id, task.name, retry_count)
    try:
        await fire_push(
            _app,
            interface=task.interface,
            thread_id=task.thread_id,
            message=task.message,
            push_type=task.push_type,
        )
    except Exception:
        logger.exception(
            "fire_task: delivery failed task_id=%s name=%s attempt=%d",
            task_id,
            task.name,
            retry_count,
        )
        _schedule_task_retry(task_id, retry_count)


async def fire_push(
    app: FastAPI,
    *,
    interface: InterfaceName,
    thread_id: str,
    message: str,
    push_type: Literal["forward", "agent", "agent_isolated", "inject"],
    topic_name: str | None = None,
    chat_id: int | None = None,
) -> None:
    """Resolve the target thread and deliver a push message.

    Args:
        app:        The FastAPI application (for accessing shared state).
        interface:  Registered interface name, e.g. ``"telegram"``.
        thread_id:  ``"default"``, ``"new"``, or a specific thread ID.
        message:    Text to forward or use as the agent prompt.
        push_type:  Delivery mode:
                    ``"forward"``        — relay message as-is, no agent run.
                    ``"agent"``          — agent run in the thread's history.
                    ``"agent_isolated"`` — agent run in a throwaway session.
                    ``"inject"``         — write message to context + forward to chat, no reply.
        topic_name: Topic name when creating a new thread (``thread_id="new"``).
        chat_id:    Group chat ID when creating a new Telegram forum topic.

    Raises:
        ValueError: if the interface is not registered.
    """
    interfaces = getattr(app.state, "interfaces", {})
    iface = interfaces.get(interface)
    if iface is None:
        raise ValueError(f"Interface {interface!r} is not registered")

    actual_thread_id = await iface.resolve_thread(thread_id, topic_name=topic_name, chat_id=chat_id)

    if push_type == "forward":
        await iface.send_proactive(actual_thread_id, message)
        return

    if push_type == "inject":
        agent_version = _resolve_agent_version(interface, actual_thread_id)
        agent = get_agent(agent_version)
        inject_config: RunnableConfig = {
            "configurable": {
                "thread_id": actual_thread_id,
                "interface": interface,
                "sender_id": _extract_chat_id(actual_thread_id) or "",
            },
        }
        await agent.aupdate_state(
            inject_config,
            {"messages": [HumanMessage(content=message)]},
            app.state.checkpointer,
        )
        await iface.send_proactive(actual_thread_id, message)
        return

    # push_type == "agent" or "agent_isolated"
    agent_version = _resolve_agent_version(interface, actual_thread_id)
    agent = get_agent(agent_version)

    # agent_isolated uses a throw-away thread so it doesn't share history with
    # the target thread.  The orphaned checkpoints accumulate in Postgres; a
    # periodic cleanup job should purge push-isolated-* threads older than a few days.
    run_thread_id = (
        f"push-isolated-{uuid.uuid4()}" if push_type == "agent_isolated" else actual_thread_id
    )

    config: RunnableConfig = {
        "recursion_limit": _PUSH_RECURSION_LIMIT,
        "configurable": {
            "thread_id": run_thread_id,
            "interface": interface,
            "sender_id": _extract_chat_id(actual_thread_id) or "",
        },
    }

    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        thread_id=run_thread_id,
        interface=interface,
    )
    stream = _push_agent_stream(agent, initial_state, config, app.state.checkpointer)
    try:
        await iface.send_proactive_stream(actual_thread_id, stream)
    except GraphRecursionError:
        logger.warning(
            "fire_push: recursion limit hit interface=%s thread_id=%s", interface, actual_thread_id
        )
        await iface.send_proactive(
            actual_thread_id,
            "⚠️ Reached the step limit without completing — the task may be too complex.",
        )


def _schedule_task_retry(task_id: str, retry_count: int) -> None:
    """Schedule a retry fire_task job with exponential backoff, or dead-letter."""
    if retry_count >= _MAX_TASK_RETRIES:
        logger.error(
            "fire_task: dead-lettered task_id=%s after %d retries", task_id, _MAX_TASK_RETRIES
        )
        return
    delay = min(_RETRY_DELAY_CAP_MINUTES, 2**retry_count)
    retry_at = datetime.now(UTC) + timedelta(minutes=delay)
    next_attempt = retry_count + 1
    job_id = f"{TASK_RETRY_JOB_PREFIX}{task_id}-{next_attempt}"
    _app.state.scheduler.add_job(  # type: ignore[union-attr]
        fire_task,
        "date",
        run_date=retry_at,
        args=[task_id, next_attempt],
        id=job_id,
        replace_existing=True,
    )
    logger.warning(
        "fire_task: retry scheduled task_id=%s attempt=%d/%d retry_at=%s",
        task_id,
        next_attempt,
        _MAX_TASK_RETRIES,
        retry_at.isoformat(),
    )


async def _push_agent_stream(
    agent: BaseAgent,
    initial_state: AgentState,
    config: RunnableConfig,
    checkpointer,
) -> AsyncIterator[AgentEvent]:
    """Loop across interrupt_after=["call_tools"] boundaries, yielding all events.

    Mirrors the logic in base.py:_agent_stream but without stop/injection handling,
    since push runs have no interactive user present.
    """
    graph_input: AgentState | Command | None = initial_state
    while True:
        async for event in agent.astream_events(graph_input, config, checkpointer):
            yield event

        graph_state = await agent.aget_state(config, checkpointer)
        if not graph_state.next:
            return  # graph reached END naturally
        if graph_state.interrupts:
            return  # approval interrupt — skip silently in push context
        graph_input = None  # resume from where the graph paused


def _resolve_agent_version(interface: InterfaceName, thread_id: str) -> str:
    """Look up the agent version configured for the given thread."""
    if interface == "telegram":
        chat_id = _extract_chat_id(thread_id)
        if chat_id:
            return load_settings().telegram.chats.get(chat_id, TelegramChatSettings()).agent
    return "v10_claude"


def _extract_chat_id(thread_id: str) -> str | None:
    """Extract the chat_id string from a ``tg-{chat_id}-…`` thread ID."""
    m = _TG_CHAT_ID_RE.match(thread_id)
    return m.group(1) if m else None
