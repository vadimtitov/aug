"""Tests for aug.utils.tasks — ScheduledTask CRUD.

Behaviors under test:
  - create_task returns a UUID string
  - get_task returns ScheduledTask when found, None when missing
  - list_tasks returns all tasks in creation order
  - update_task returns True when found, False when missing
  - update_task rejects immutable fields and no-op calls
  - delete_task returns True when found, False when missing
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from aug.utils.tasks import (
    ScheduledTask,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    update_task,
)

_NOW = datetime(2026, 5, 17, 10, 0, tzinfo=UTC)
_CRON_PARAMS = {"hour": 10, "minute": 0}


def _task_row(**overrides):
    row = {
        "id": "uuid-1",
        "name": "morning-news",
        "interface": "telegram",
        "thread_id": "default",
        "message": "search for news",
        "schedule_type": "cron",
        "schedule_params": _CRON_PARAMS,
        "enabled": True,
        "push_type": "agent",
        "created_at": _NOW,
    }
    row.update(overrides)
    return row


# ── create_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_task_returns_uuid():
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value="uuid-abc")

    result = await create_task(
        conn,
        name="morning-news",
        interface="telegram",
        thread_id="default",
        message="search for news",
        schedule_type="cron",
        schedule_params=_CRON_PARAMS,
    )

    assert result == "uuid-abc"
    conn.fetchval.assert_called_once()


# ── get_task ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_task_returns_task():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=_task_row())

    task = await get_task(conn, "uuid-1")

    assert isinstance(task, ScheduledTask)
    assert task.id == "uuid-1"
    assert task.name == "morning-news"
    assert task.thread_id == "default"
    assert task.schedule_params == _CRON_PARAMS


@pytest.mark.asyncio
async def test_get_task_missing_returns_none():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)

    result = await get_task(conn, "missing-id")

    assert result is None


# ── list_tasks ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tasks_returns_all():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[_task_row(), _task_row(id="uuid-2", name="evening-check")])

    tasks = await list_tasks(conn)

    assert len(tasks) == 2
    assert tasks[0].name == "morning-news"
    assert tasks[1].name == "evening-check"


@pytest.mark.asyncio
async def test_list_tasks_empty():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])

    tasks = await list_tasks(conn)

    assert tasks == []


# ── update_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_returns_true_when_found():
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value="uuid-1")

    result = await update_task(conn, "uuid-1", message="updated message")

    assert result is True
    conn.fetchval.assert_called_once()


@pytest.mark.asyncio
async def test_update_task_returns_false_when_missing():
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)

    result = await update_task(conn, "missing-id", message="updated")

    assert result is False


@pytest.mark.asyncio
async def test_update_task_rejects_immutable_fields():
    conn = AsyncMock()

    with pytest.raises(ValueError, match="Cannot update"):
        await update_task(conn, "uuid-1", id="new-id")


@pytest.mark.asyncio
async def test_update_task_no_fields_returns_false_without_db_call():
    conn = AsyncMock()

    result = await update_task(conn, "uuid-1")

    assert result is False
    conn.fetchval.assert_not_called()


# ── delete_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_task_returns_true_when_found():
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value="uuid-1")

    result = await delete_task(conn, "uuid-1")

    assert result is True


@pytest.mark.asyncio
async def test_delete_task_returns_false_when_missing():
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)

    result = await delete_task(conn, "missing-id")

    assert result is False
