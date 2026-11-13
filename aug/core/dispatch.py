"""Proactive delivery engine for scheduled tasks and external pushes.

Three public entry points:

fire_task(task_id)
    Called by APScheduler when a scheduled task fires.  Loads the task from
    the database, then delegates to fire_push.

fire_push(app, *, interface, context_id, message, push_type, isolated, ...)
    Resolves the target thread via the interface, then either:
      - delivers the message directly (push_type="forward"), or
      - runs a constrained agent turn and delivers the final response
        (push_type="agent").

broadcast(app, message)
    Delivers a service-level announcement to every registered interface, with
    no target thread and no agent run.

Agent pushes run through the harness like any turn, under the same run lock, as an
unattended turn (``TurnContext.attended=False``), which the harness caps at a few steps.
"""

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import FastAPI

from aug.core.harness import (
    Address,
    HarnessEvent,
    Message,
    Sender,
    TextPart,
    TurnContext,
    TurnEnded,
    error_text,
)
from aug.core.harness.langgraph import LangGraphHarness
from aug.core.prompts import UNATTENDED_TIMED_OUT_MESSAGE
from aug.core.registry import get_harness
from aug.core.run import run_registry
from aug.utils.app_ref import get_app
from aug.utils.db import get_pool
from aug.utils.tasks import ScheduledTask, get_task, mark_fired

InterfaceName = Literal["telegram", "rest_api"]


class PushTurnFailed(RuntimeError):
    """An agent push ran but its turn ended with an error."""


logger = logging.getLogger(__name__)

# How long an agent push or scheduled task may run.  Nobody is watching it and it can't
# be stopped with /stop (an isolated push has a context of its own), so a command that
# hangs would otherwise hold the chat's run lock, and the task's scheduler slot, for good.
UNATTENDED_TIMEOUT_SECONDS = 15 * 60

# Retry configuration for fire_task delivery failures.
_MAX_TASK_RETRIES = 10
_RETRY_DELAY_CAP_MINUTES = 60
# Prefix used for APScheduler retry job IDs — must not collide with task UUIDs.
TASK_RETRY_JOB_PREFIX = "retry-"

# Regex to extract the chat_id portion from a tg-{chat_id}-… thread ID.
_TG_CHAT_ID_RE = re.compile(r"^tg-(-?\d+)-")


async def fire_task(task_id: str, retry_count: int = 0) -> None:
    """Entry point called by APScheduler when a scheduled task fires.

    Loads the task from the database, skips if disabled, then delegates to
    fire_push.  On delivery failure, schedules an exponential-backoff retry job
    (up to _MAX_TASK_RETRIES attempts) before dead-lettering.
    """
    app = get_app()
    if app is None:
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
            app,
            interface=task.interface,
            context_id=task.thread_id,  # the tasks table keeps its column name
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
        return

    await _record_one_shot_fired(task)


async def fire_push(
    app: FastAPI,
    *,
    interface: InterfaceName,
    context_id: str,
    message: str,
    push_type: Literal["forward", "agent", "agent_isolated", "inject"],
    topic_name: str | None = None,
    chat_id: int | None = None,
) -> None:
    """Resolve the target thread and deliver a push message.

    Args:
        app:        The FastAPI application (for accessing shared state).
        interface:  Registered interface name, e.g. ``"telegram"``.
        context_id:  ``"default"``, ``"new"``, or a specific thread ID.
        message:    Text to forward or use as the agent prompt.
        push_type:  Delivery mode:
                    ``"forward"``        — relay message as-is, no agent run.
                    ``"agent"``          — agent run in the thread's history.
                    ``"agent_isolated"`` — agent run in a throwaway session.
                    ``"inject"``         — write message to context + forward to chat, no reply.
        topic_name: Topic name when creating a new thread (``context_id="new"``).
        chat_id:    Group chat ID when creating a new Telegram forum topic.

    Raises:
        ValueError: if the interface is not registered.
    """
    interfaces = getattr(app.state, "interfaces", {})
    iface = interfaces.get(interface)
    if iface is None:
        raise ValueError(f"Interface {interface!r} is not registered")

    actual_context_id = await iface.resolve_context(
        context_id, topic_name=topic_name, chat_id=chat_id
    )

    if push_type == "forward":
        await iface.send_proactive(actual_context_id, message)
        return

    harness = get_harness(iface.get_agent_version(actual_context_id))
    if push_type == "inject":
        if not isinstance(harness, LangGraphHarness):
            raise ValueError(f"inject pushes are not supported on {type(harness).__name__}")
        await harness.add_context(message, actual_context_id)
        await iface.send_proactive(actual_context_id, message)
        return

    # push_type == "agent" or "agent_isolated"
    # agent_isolated uses a throw-away session so it doesn't share history with the
    # target thread.  The orphaned checkpoints accumulate in Postgres; a periodic
    # cleanup job should purge push-isolated-* threads older than a few days.
    run_context_id = (
        f"push-isolated-{uuid.uuid4()}" if push_type == "agent_isolated" else actual_context_id
    )
    turn = TurnContext(
        reply_to=Address(
            interface=interface,
            conversation_id=iface.conversation_id(actual_context_id),
            chat_id=_extract_chat_id(actual_context_id) or "",
        ),
        attended=False,
    )
    prompt = Message(parts=[TextPart(text=message)], sender=Sender(interface=interface, user_id=""))
    ended: TurnEnded | None = None

    async def events(stream: AsyncIterator[HarnessEvent]) -> AsyncIterator[HarnessEvent]:
        nonlocal ended
        async for event in stream:
            if isinstance(event, TurnEnded):
                ended = event
                if event.error is not None and event.error.kind != "step_limit":
                    # Raised inside the stream, before the interface sends the text it
                    # collected: fire_task retries the whole turn, so a partial reply
                    # now would reach the user twice.
                    raise PushTurnFailed(f"push turn failed: {event.error.kind}")
            yield event

    async with run_registry.run_lock(run_context_id):
        timeout = asyncio.timeout(UNATTENDED_TIMEOUT_SECONDS)
        try:
            async with timeout:
                async with aclosing(harness.run(prompt, run_context_id, turn)) as stream:
                    await iface.send_proactive_stream(actual_context_id, events(stream))
        except TimeoutError:
            if not timeout.expired():
                raise  # a timeout inside delivery, not ours: fire_task retries it
            # Not retried: the same turn would most likely hang the same way.
            logger.warning("push turn timed out context=%s", run_context_id)
            minutes = UNATTENDED_TIMEOUT_SECONDS // 60
            await iface.send_proactive(
                actual_context_id, UNATTENDED_TIMED_OUT_MESSAGE.format(minutes=minutes)
            )
            return

    if ended is not None and ended.error is not None:  # step_limit: retrying won't help
        await iface.send_proactive(actual_context_id, error_text(ended.error))


async def deliver_unsolicited(context_id: str, events: AsyncIterator[HarnessEvent]) -> None:
    """Receive a turn a harness started by itself (the ``running()`` sink).

    LangGraph never starts one; the Claude Code harness will (background work that
    finishes after its turn) and routes it here.  Until then there is nothing to
    deliver to, so such a turn is drained and logged rather than lost silently.
    """
    async for _ in events:
        pass
    logger.warning("unsolicited turn dropped context=%s: no delivery route yet", context_id)


async def broadcast(app: FastAPI, message: str) -> int:
    """Deliver *message* to every announcement thread of every registered interface.

    The caller says what happened; each interface decides who hears it (see
    ``BaseInterface.announcement_threads``), so nothing here knows that Telegram
    exists.  An interface with no push channel simply reports no threads.

    Best-effort: a failing interface or thread is logged and skipped, never
    raised, so one unreachable chat cannot take down whatever triggered the
    broadcast.

    Returns the number of threads the message reached.
    """
    delivered = 0
    for name, iface in getattr(app.state, "interfaces", {}).items():
        try:
            threads = await iface.announcement_threads()
        except Exception:
            logger.warning("broadcast: target lookup failed interface=%s", name, exc_info=True)
            continue
        for context_id in threads:
            try:
                await iface.send_proactive(context_id, message)
                delivered += 1
            except Exception as e:
                # No traceback: a failure here is an API-level answer that says all
                # there is to say ("Chat not found" for an owner who never DM'd the
                # bot), and it happens on every boot until they do.
                logger.warning(
                    "broadcast: delivery failed interface=%s context_id=%s error=%s",
                    name,
                    context_id,
                    e,
                )
    return delivered


async def _record_one_shot_fired(task: ScheduledTask) -> None:
    """Stamp a delivered ``date`` task as finished; recurring tasks are left alone.

    This is the fact the reconciler reads to know the task is over.  Without it the
    row looks pending for ever, and gets re-registered with APScheduler every 30
    seconds only to misfire.

    Failure is logged and swallowed: the message is already delivered, and a missing
    stamp is repaired on the next reconcile pass, which marks past-due one-shots itself.
    """
    if task.schedule_type != "date":
        return
    try:
        async with get_pool().acquire() as conn:
            await mark_fired(conn, task.id)
        logger.info("fire_task: one-shot complete task_id=%s name=%s", task.id, task.name)
    except Exception:
        logger.exception("fire_task: could not stamp one-shot task_id=%s", task.id)


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
    get_app().state.scheduler.add_job(  # type: ignore[union-attr]
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


def _extract_chat_id(context_id: str) -> str | None:
    """Extract the chat_id string from a ``tg-{chat_id}-…`` thread ID."""
    m = _TG_CHAT_ID_RE.match(context_id)
    return m.group(1) if m else None
