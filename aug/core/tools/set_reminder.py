"""Reminder tool — schedules a message to be delivered to the user at a future time."""

import logging
from datetime import UTC, datetime

import asyncpg
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aug.config import get_settings
from aug.utils.db import strip_driver
from aug.utils.reminders import create_reminder
from aug.utils.user_settings import get_setting

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

    thread_id = (config.get("configurable") or {}).get("thread_id", "")
    notification_interface = get_setting("thread_notifications", thread_id, "interface", default="")
    notification_target = get_setting("thread_notifications", thread_id, "id", default="")

    dsn = strip_driver(get_settings().DATABASE_URL)
    try:
        conn = await asyncpg.connect(dsn)
        try:
            reminder_id = await create_reminder(
                conn, message, trigger_at, notification_interface, notification_target
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.error("set_reminder DB error: %s", e)
        return f"Failed to save reminder: {e}"

    logger.info(
        "set_reminder id=%s trigger_at=%s interface=%s target=%s",
        reminder_id,
        trigger_at.isoformat(),
        notification_interface,
        notification_target,
    )
    delta = trigger_at - now
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    human = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    return f"Reminder set for {trigger_at.strftime('%Y-%m-%d %H:%M %Z')} (in {human}): {message!r}"
