"""Reminder tool — schedules a message to be delivered to the user at a future time."""

import logging
import uuid
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aug.utils.db import get_pool
from aug.utils.tasks import create_task

logger = logging.getLogger(__name__)


@tool
async def set_reminder(when: str, message: str, config: RunnableConfig) -> str:
    """Schedule a reminder to be delivered to you at a specific time via Telegram.

    Reminders survive server restarts and are guaranteed to fire.

    Use get_current_datetime first to know the current time, then compute the
    target datetime and pass it as an ISO 8601 string.

    Args:
        when: ISO 8601 datetime string with timezone, e.g. "2026-03-18T09:00:00+00:00".
        message: The reminder text to deliver.
    """
    try:
        trigger_at = datetime.fromisoformat(when)
        if trigger_at.tzinfo is None:
            trigger_at = trigger_at.replace(tzinfo=UTC)
    except ValueError as e:
        return f"Invalid datetime {when!r}: {e}. Use ISO 8601, e.g. 2026-03-18T09:00:00+00:00"

    now = datetime.now(tz=UTC)
    if trigger_at <= now:
        return f"Reminder time {when!r} is in the past. Please provide a future datetime."

    configurable = config.get("configurable") or {}
    interface: str = configurable.get("interface", "telegram")
    sender_id: str = configurable.get("sender_id", "")

    if interface == "telegram" and sender_id:
        thread_id = f"default:{sender_id}"
    else:
        thread_id = "default"

    name = f"reminder-{uuid.uuid4().hex[:8]}"
    full_message = f"⏰ {message}"

    try:
        async with get_pool().acquire() as conn:
            task_id = await create_task(
                conn,
                name=name,
                interface=interface,
                thread_id=thread_id,
                message=full_message,
                schedule_type="date",
                schedule_params={"run_date": trigger_at.isoformat()},
                push_type="forward",
            )
    except Exception as e:
        logger.error("set_reminder DB error: %s", e)
        return f"Failed to save reminder: {e}"

    logger.info(
        "set_reminder task_id=%s trigger_at=%s interface=%s thread_id=%s",
        task_id,
        trigger_at.isoformat(),
        interface,
        thread_id,
    )
    delta = trigger_at - now
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    human = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    return f"Reminder set for {trigger_at.strftime('%Y-%m-%d %H:%M %Z')} (in {human}): {message!r}"
