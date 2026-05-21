"""Agent tools for managing scheduled tasks.

Four tools are exposed to the LLM:

  create_task  — schedule a new recurring or one-shot task
  list_tasks   — show all tasks
  update_task  — modify an existing task by name
  delete_task  — remove a task by name

All tools write directly to the database.  The scheduler reconciler picks up
changes within its next poll cycle (≤ 30 s).

Thread routing at creation time:
  - Telegram DM thread (``tg-{id}-{session}``)  → stored as ``"default:{chat_id}"``
    so that a future ``/clear`` (which increments the session counter) doesn't orphan
    the task, while still routing deterministically to the right chat.
  - Telegram topic thread (``tg-{id}-topic-{n}``) → stored as-is (topics are stable).
  - Any other interface → stored as ``"default"`` (interface handles resolution).
"""

import logging
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aug.utils.db import get_pool
from aug.utils.job_control import add_task_job, cancel_job
from aug.utils.tasks import (
    PushType,
    ScheduleType,
    get_task_by_name,
    make_trigger,
)
from aug.utils.tasks import (
    create_task as _create_task,
)
from aug.utils.tasks import (
    delete_task as _delete_task,
)
from aug.utils.tasks import (
    list_tasks as _list_tasks,
)
from aug.utils.tasks import (
    update_task as _update_task,
)

logger = logging.getLogger(__name__)

_TG_CHAT_RE = re.compile(r"^tg-(-?\d+)-")


@tool
async def create_task(
    name: str,
    schedule_type: ScheduleType,
    schedule_params: dict,
    message: str,
    config: RunnableConfig,
    push_type: PushType = "agent",
) -> str:
    """Schedule a recurring or one-shot task — use this for reminders, nudges, and
    automated agent actions alike.

    Args:
        name:            Short unique identifier (e.g. "morning-news", "dentist-reminder").
                         Lowercase with hyphens, no spaces.
        schedule_type:   "cron", "interval", or "date".
        schedule_params: Trigger config dict.
                         Cron:     {"hour": 18, "minute": 0, "timezone": "Europe/Berlin"}
                         Interval: {"hours": 6}  or  {"minutes": 30}
                         Date:     {"run_date": "2026-06-01T09:00:00+00:00"}
        message:         Text sent or used as the agent prompt when the task fires.
        push_type:       Delivery mode (default: "agent"):
                         "agent"          — agent runs in this thread and replies. Best for
                                           tasks where a response is expected (news summary,
                                           status report, smart reminder with context).
                         "forward"        — message delivered as-is, no agent reply. Best for
                                           simple text reminders ("⏰ take your meds").
                         "inject"         — message shown in chat and added to context, no
                                           reply. Best for nudges you may want to respond to
                                           ("are you ready to leave work?").
                         "agent_isolated" — agent runs in a fresh session, no shared history.
    """
    configurable = config.get("configurable") or {}
    raw_thread_id: str = configurable.get("thread_id", "")
    interface: str = configurable.get("interface", "telegram")

    if interface == "telegram":
        if "-topic-" not in raw_thread_id:
            m = _TG_CHAT_RE.match(raw_thread_id)
            # Embed the chat_id so resolve_thread routes deterministically after /clear.
            thread_id = f"default:{m.group(1)}" if m else "default"
        else:
            thread_id = raw_thread_id
    else:
        thread_id = "default"

    try:
        async with get_pool().acquire() as conn:
            task_id = await _create_task(
                conn,
                name=name,
                interface=interface,
                thread_id=thread_id,
                message=message,
                schedule_type=schedule_type,
                schedule_params=schedule_params,
                push_type=push_type,
            )
    except Exception as exc:
        logger.error("create_task DB error: %s", exc)
        return f"Failed to create task: {exc}"

    # Register immediately so short-duration tasks (e.g. 10-second reminders) aren't
    # missed waiting for the 30-second reconciler cycle.
    try:
        trigger = make_trigger(schedule_type, schedule_params)
        add_task_job(task_id, trigger)
    except Exception as exc:
        logger.warning("add_task_job failed (reconciler will pick it up): %s", exc)

    schedule_desc = _describe_schedule(schedule_type, schedule_params)
    return (
        f"Task '{name}' created (id: {task_id[:8]}…). "
        f"Schedule: {schedule_desc}. "
        f"It will run in this thread."
    )


@tool
async def list_tasks(config: RunnableConfig) -> str:
    """List all scheduled tasks with their name, schedule, and delivery thread."""
    try:
        async with get_pool().acquire() as conn:
            tasks = await _list_tasks(conn)
    except Exception as exc:
        logger.error("list_tasks DB error: %s", exc)
        return f"Failed to list tasks: {exc}"

    if not tasks:
        return "No scheduled tasks."

    lines = ["Scheduled tasks:\n"]
    for t in tasks:
        status = "enabled" if t.enabled else "disabled"
        schedule = _describe_schedule(t.schedule_type, t.schedule_params)
        lines.append(
            f"• {t.name} (id: {t.id[:8]}…)  [{status}]\n"
            f"  Schedule: {schedule}\n"
            f"  Thread: {t.thread_id}\n"
            f"  Message: {t.message!r}"
        )
    return "\n".join(lines)


@tool
async def update_task(
    name: str,
    message: str | None = None,
    schedule_type: ScheduleType | None = None,
    schedule_params: dict | None = None,
    enabled: bool | None = None,
    push_type: PushType | None = None,
    config: RunnableConfig = None,
) -> str:
    """Update an existing scheduled task or reminder identified by name.

    Only provide the fields you want to change.  At least one field must be given.

    Args:
        name:            The task name to update.
        message:         New prompt message (optional).
        schedule_type:   New schedule type: "cron", "interval", or "date" (optional).
        schedule_params: New schedule parameters dict (optional).
        enabled:         Set to False to pause without deleting (optional).
        push_type:       Change delivery mode: "agent", "forward", "inject",
                         or "agent_isolated" (optional).
    """
    fields: dict = {}
    if message is not None:
        fields["message"] = message
    if schedule_type is not None:
        fields["schedule_type"] = schedule_type
    if schedule_params is not None:
        fields["schedule_params"] = schedule_params
    if enabled is not None:
        fields["enabled"] = enabled
    if push_type is not None:
        fields["push_type"] = push_type

    if not fields:
        return (
            "No fields to update — provide at least one of: "
            "message, schedule_type, schedule_params, enabled, push_type."
        )

    try:
        async with get_pool().acquire() as conn:
            task = await get_task_by_name(conn, name)
            if task is None:
                return f"Task '{name}' not found."
            updated = await _update_task(conn, task.id, **fields)
    except Exception as exc:
        logger.error("update_task DB error: %s", exc)
        return f"Failed to update task '{name}': {exc}"

    if not updated:
        return f"Task '{name}' could not be updated — it may have been deleted concurrently."
    changed = ", ".join(fields.keys())
    return f"Task '{name}' updated ({changed})."


@tool
async def delete_task(name: str, config: RunnableConfig) -> str:
    """Permanently delete a scheduled task by name.

    Args:
        name: The task name to delete.
    """
    try:
        async with get_pool().acquire() as conn:
            task = await get_task_by_name(conn, name)
            if task is None:
                return f"Task '{name}' not found."
            await _delete_task(conn, task.id)
    except Exception as exc:
        logger.error("delete_task DB error: %s", exc)
        return f"Failed to delete task '{name}': {exc}"

    cancel_job(task.id)
    return f"Task '{name}' deleted."


def _describe_schedule(schedule_type: ScheduleType, params: dict) -> str:
    """Return a short human-readable description of a schedule."""
    match schedule_type:
        case "cron":
            tz = params.get("timezone", "UTC")
            parts = [f"{k}={v}" for k, v in params.items() if k != "timezone"]
            return f"cron ({', '.join(parts)}, tz={tz})"
        case "interval":
            parts = [f"{v} {k}" for k, v in params.items()]
            return f"every {', '.join(parts)}"
        case "date":
            return f"once at {params.get('run_date', '?')}"
        case _:
            return f"{schedule_type} {params}"
