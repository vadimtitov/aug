"""Scheduled task CRUD — authoritative store for all task definitions.

Reads and writes to the ``scheduled_tasks`` Postgres table.  The APScheduler
instance is loaded from this table on startup and re-synced by a background
reconciler every 30 seconds, so changes made via the agent tools are picked up
without a restart.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import asyncpg
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

ScheduleType = Literal["cron", "interval", "date"]
PushType = Literal["forward", "agent", "agent_isolated", "inject"]

_UPDATABLE = frozenset(
    {"name", "message", "schedule_type", "schedule_params", "enabled", "thread_id", "push_type"}
)


@dataclass(frozen=True)
class ScheduledTask:
    """A persistent task that fires an agent message on a schedule.

    Attributes:
        id:              UUID primary key.
        name:            Short unique human-readable identifier (e.g. ``"morning-news"``).
        interface:       Delivery interface, e.g. ``"telegram"``.
        thread_id:       Target thread — ``"default"``, or a specific thread ID such as
                         ``"tg-12345-topic-67"``.
        message:         Prompt text injected as a user message when the task fires.
        schedule_type:   One of ``"cron"``, ``"interval"``, or ``"date"``.
        schedule_params: APScheduler trigger kwargs.
                         Cron example: ``{"hour": 10, "minute": 0, "timezone": "Europe/Berlin"}``.
                         Interval example: ``{"minutes": 30}``.
                         Date example: ``{"run_date": "2026-06-01T09:00:00+00:00"}``.
        enabled:         Whether the task is currently active.
        created_at:      UTC creation timestamp.
    """

    id: str
    name: str
    interface: str
    thread_id: str
    message: str
    schedule_type: ScheduleType
    schedule_params: dict
    enabled: bool
    push_type: str
    created_at: datetime


async def create_task(
    conn: asyncpg.Connection,
    *,
    name: str,
    interface: str,
    thread_id: str,
    message: str,
    schedule_type: ScheduleType,
    schedule_params: dict,
    push_type: str = "agent",
) -> str:
    """Insert a new scheduled task and return its UUID."""
    return str(
        await conn.fetchval(
            """
            INSERT INTO scheduled_tasks
                (name, interface, thread_id, message, schedule_type, schedule_params, push_type)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING id
            """,
            name,
            interface,
            thread_id,
            message,
            schedule_type,
            json.dumps(schedule_params),
            push_type,
        )
    )


async def get_task(conn: asyncpg.Connection, task_id: str) -> ScheduledTask | None:
    """Return the task with ``task_id``, or ``None`` if not found."""
    row = await conn.fetchrow("SELECT * FROM scheduled_tasks WHERE id = $1", task_id)
    return _row_to_task(row) if row else None


async def get_task_by_name(conn: asyncpg.Connection, name: str) -> ScheduledTask | None:
    """Return the task with ``name``, or ``None`` if not found."""
    row = await conn.fetchrow("SELECT * FROM scheduled_tasks WHERE name = $1", name)
    return _row_to_task(row) if row else None


async def list_tasks(conn: asyncpg.Connection) -> list[ScheduledTask]:
    """Return all tasks ordered by creation time."""
    rows = await conn.fetch("SELECT * FROM scheduled_tasks ORDER BY created_at")
    return [_row_to_task(r) for r in rows]


async def update_task(conn: asyncpg.Connection, task_id: str, **fields) -> bool:
    """Update allowed fields on a task.

    Returns:
        ``True`` if the task was found and updated, ``False`` if not found.

    Raises:
        ValueError: if any key in *fields* is not an updatable column.
    """
    invalid = set(fields) - _UPDATABLE
    if invalid:
        raise ValueError(f"Cannot update fields: {sorted(invalid)}")
    if not fields:
        return False

    set_parts = []
    values = []
    for i, (col, val) in enumerate(fields.items(), start=1):
        if col == "schedule_params":
            set_parts.append(f"{col} = ${i}::jsonb")
            values.append(json.dumps(val))
        else:
            set_parts.append(f"{col} = ${i}")
            values.append(val)

    values.append(task_id)
    sql = (
        f"UPDATE scheduled_tasks SET {', '.join(set_parts)} WHERE id = ${len(values)} RETURNING id"
    )
    row = await conn.fetchval(sql, *values)
    return row is not None


def make_trigger(schedule_type: str, schedule_params: dict):
    """Build the correct APScheduler trigger from schedule fields."""
    params = schedule_params.copy()
    match schedule_type:
        case "cron":
            return CronTrigger(**params)
        case "interval":
            return IntervalTrigger(**params)
        case "date":
            if "run_date" in params:
                params["run_date"] = _normalize_run_date(params["run_date"])
            return DateTrigger(**params)
        case _:
            raise ValueError(f"Unknown schedule_type: {schedule_type!r}")


async def delete_task(conn: asyncpg.Connection, task_id: str) -> bool:
    """Delete a task by ID.

    Returns:
        ``True`` if the task existed and was deleted, ``False`` if not found.
    """
    row = await conn.fetchval("DELETE FROM scheduled_tasks WHERE id = $1 RETURNING id", task_id)
    return row is not None


def _normalize_run_date(run_date: str | datetime) -> datetime:
    """Parse a run_date value into a datetime APScheduler can consume.

    Handles PostgreSQL's ``::text`` timestamp format (e.g. ``"2026-05-21 23:38:00+00"``)
    which uses a space separator and a truncated timezone offset lacking the ``:MM`` part
    that APScheduler's own parser requires.
    """
    if isinstance(run_date, datetime):
        return run_date
    # Normalise: space separator → T, +HH (no minutes) → +HH:00
    s = run_date.replace(" ", "T")
    s = re.sub(r"([+-]\d{2})$", r"\1:00", s)
    return datetime.fromisoformat(s)


def _row_to_task(row) -> ScheduledTask:
    params = row["schedule_params"]
    if isinstance(params, str):
        params = json.loads(params)
    return ScheduledTask(
        id=str(row["id"]),
        name=row["name"],
        interface=row["interface"],
        thread_id=row["thread_id"],
        message=row["message"],
        schedule_type=row["schedule_type"],
        schedule_params=dict(params) if params else {},
        enabled=row["enabled"],
        push_type=row["push_type"],
        created_at=row["created_at"],
    )
