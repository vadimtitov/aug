"""Reminder storage and delivery loop.

Polls the ``reminders`` table every 30 seconds and delivers due reminders via
the stored notification interface.  A reminder is only marked ``fired=TRUE``
after successful delivery.  On failure, exponential back-off is applied:

    delay = min(60, 2 ** retry_count) minutes   (1 → 2 → 4 → … → 60 min)

After _MAX_RETRIES failed attempts the reminder is dead-lettered: marked with
``dead_lettered_at`` and never retried again.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import asyncpg
from fastapi import FastAPI

from aug.utils.notify import send_notification

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30  # seconds
_MAX_BACKOFF_MINUTES = 60
_MAX_RETRIES = 50


async def create_reminder(
    conn: asyncpg.Connection,
    message: str,
    trigger_at: datetime,
    notification_interface: str,
    notification_target: str,
) -> str:
    """Insert a reminder row and return its UUID."""
    return await conn.fetchval(
        """INSERT INTO reminders (message, trigger_at, notification_interface, notification_target)
           VALUES ($1, $2, $3, $4) RETURNING id""",
        message,
        trigger_at,
        notification_interface,
        notification_target,
    )


def start_reminder_loop(app: FastAPI) -> asyncio.Task:
    """Start the background reminder loop and return the task."""
    return asyncio.create_task(_reminder_loop(app))


async def _reminder_loop(app: FastAPI) -> None:
    try:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                await _fire_due_reminders(app)
            except Exception:
                logger.exception("Reminder loop error")
    except asyncio.CancelledError:
        logger.info("Reminder loop shut down cleanly.")


async def _fire_due_reminders(app: FastAPI) -> None:
    pool = app.state.db_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, message, notification_interface, notification_target, retry_count
            FROM reminders
            WHERE fired = FALSE
              AND dead_lettered_at IS NULL
              AND trigger_at <= NOW()
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
            FOR UPDATE SKIP LOCKED
            """
        )

    logger.debug("reminder_check pending=%d", len(rows))
    for row in rows:
        reminder_id = row["id"]
        retry_count = row["retry_count"]
        try:
            await send_notification(
                app,
                row["notification_interface"],
                row["notification_target"],
                f"\u23f0 {row['message']}",
            )
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE reminders SET fired = TRUE WHERE id = $1",
                    reminder_id,
                )
            logger.info(
                "Reminder delivered id=%s after %d attempt(s)", reminder_id, retry_count + 1
            )
        except Exception as exc:
            new_retry_count = retry_count + 1
            error_text = str(exc)[:500]
            if new_retry_count >= _MAX_RETRIES:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE reminders
                        SET retry_count      = $1,
                            last_error       = $2,
                            dead_lettered_at = NOW()
                        WHERE id = $3
                        """,
                        new_retry_count,
                        error_text,
                        reminder_id,
                    )
                logger.error(
                    "Reminder dead-lettered id=%s after %d attempts — last error: %s",
                    reminder_id,
                    new_retry_count,
                    error_text,
                )
            else:
                delay = _next_retry_delay(retry_count)
                next_retry = datetime.now(tz=UTC) + delay
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE reminders
                        SET retry_count   = retry_count + 1,
                            next_retry_at = $1,
                            last_error    = $2
                        WHERE id = $3
                        """,
                        next_retry,
                        error_text,
                        reminder_id,
                    )
                logger.warning(
                    "Reminder delivery failed id=%s retry=%d next_retry_in=%s error=%r",
                    reminder_id,
                    new_retry_count,
                    delay,
                    error_text,
                )

    app.state.last_reminder_check = datetime.now(tz=UTC)


def _next_retry_delay(retry_count: int) -> timedelta:
    minutes = min(_MAX_BACKOFF_MINUTES, 2**retry_count)
    return timedelta(minutes=minutes)
