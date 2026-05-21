"""Tests for aug.core.tools.tasks — agent-facing task management tools.

Behaviors under test:
  - create_task: DM thread_id → stores "default:{chat_id}"; topic thread_id → stores as-is
  - create_task: DB error → returns error string, does not raise
  - list_tasks: no tasks → returns informative message
  - list_tasks: tasks exist → returns formatted table
  - update_task: task found → confirms update
  - update_task: task not found → returns not-found message
  - delete_task: task found → confirms deletion
  - delete_task: task not found → returns not-found message
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_DM_CONFIG = {
    "configurable": {
        "thread_id": "tg-123456-0",
        "interface": "telegram",
        "sender_id": "123456",
    }
}

_TOPIC_CONFIG = {
    "configurable": {
        "thread_id": "tg-123456-topic-99",
        "interface": "telegram",
        "sender_id": "123456",
    }
}

_CRON_PARAMS = {"hour": 10, "minute": 0}
_NOW = datetime(2026, 5, 17, 10, 0, tzinfo=UTC)


def _task_row(**overrides):
    row = {
        "id": "uuid-1",
        "name": "morning-news",
        "interface": "telegram",
        "thread_id": "default:123456",
        "message": "search for news",
        "schedule_type": "cron",
        "schedule_params": _CRON_PARAMS,
        "enabled": True,
        "push_type": "agent",
        "created_at": _NOW,
    }
    row.update(overrides)
    return row


_UNSET = object()


def _make_conn(fetchval=_UNSET, fetchrow=_UNSET, fetch=_UNSET):
    conn = AsyncMock()
    if fetchval is not _UNSET:
        conn.fetchval = AsyncMock(return_value=fetchval)
    if fetchrow is not _UNSET:
        conn.fetchrow = AsyncMock(return_value=fetchrow)
    if fetch is not _UNSET:
        conn.fetch = AsyncMock(return_value=fetch)
    return conn


def _make_pool(conn):
    """Return a mock asyncpg pool whose acquire() context manager yields conn."""
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool


# ── create_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_task_dm_thread_stores_default_with_chat_id():
    """A task created from a DM thread stores 'default:{chat_id}' as thread_id."""
    from aug.core.tools.tasks import create_task

    conn = _make_conn(fetchval="uuid-abc")

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await create_task.ainvoke(
            {
                "name": "morning-news",
                "schedule_type": "cron",
                "schedule_params": _CRON_PARAMS,
                "message": "search for news",
            },
            config=_DM_CONFIG,
        )

    assert "morning-news" in result
    call_args = conn.fetchval.call_args
    assert "default:123456" in call_args.args


@pytest.mark.asyncio
async def test_create_task_topic_thread_stores_specific_id():
    """A task created from a topic thread stores the exact thread_id."""
    from aug.core.tools.tasks import create_task

    conn = _make_conn(fetchval="uuid-abc")

    with patch("aug.utils.db._pool", _make_pool(conn)):
        await create_task.ainvoke(
            {
                "name": "topic-task",
                "schedule_type": "cron",
                "schedule_params": _CRON_PARAMS,
                "message": "do something",
            },
            config=_TOPIC_CONFIG,
        )

    call_args = conn.fetchval.call_args
    assert "tg-123456-topic-99" in call_args.args


@pytest.mark.asyncio
async def test_create_task_db_error_returns_error_string():
    """A database error should return an explicit failure message."""
    from aug.core.tools.tasks import create_task

    with patch("aug.core.tools.tasks.get_pool", side_effect=RuntimeError("pool not ready")):
        result = await create_task.ainvoke(
            {
                "name": "broken-task",
                "schedule_type": "cron",
                "schedule_params": _CRON_PARAMS,
                "message": "will fail",
            },
            config=_DM_CONFIG,
        )

    assert "failed" in result.lower()


# ── list_tasks ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tasks_empty_returns_informative_message():
    from aug.core.tools.tasks import list_tasks

    conn = _make_conn(fetch=[])

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await list_tasks.ainvoke({}, config=_DM_CONFIG)

    assert "no" in result.lower()


@pytest.mark.asyncio
async def test_list_tasks_returns_task_names():
    from aug.core.tools.tasks import list_tasks

    conn = _make_conn(fetch=[_task_row(), _task_row(id="uuid-2", name="evening-check")])

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await list_tasks.ainvoke({}, config=_DM_CONFIG)

    assert "morning-news" in result
    assert "evening-check" in result


# ── update_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_found_confirms_update():
    from aug.core.tools.tasks import update_task

    conn = _make_conn(fetchrow=_task_row(), fetchval="uuid-1")

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await update_task.ainvoke(
            {"name": "morning-news", "message": "fetch tech headlines"},
            config=_DM_CONFIG,
        )

    assert "updated" in result.lower() or "morning-news" in result


@pytest.mark.asyncio
async def test_update_task_concurrent_delete_returns_error():
    """If the task is deleted between SELECT and UPDATE, return an explicit error."""
    from aug.core.tools.tasks import update_task

    # fetchrow finds the task, but fetchval (the UPDATE RETURNING) returns None
    conn = _make_conn(fetchrow=_task_row(), fetchval=None)

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await update_task.ainvoke(
            {"name": "morning-news", "message": "too late"},
            config=_DM_CONFIG,
        )

    assert "could not be updated" in result.lower() or "deleted" in result.lower()


@pytest.mark.asyncio
async def test_update_task_not_found_returns_not_found_message():
    from aug.core.tools.tasks import update_task

    conn = _make_conn(fetchrow=None)

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await update_task.ainvoke(
            {"name": "nonexistent", "message": "something"},
            config=_DM_CONFIG,
        )

    assert "not found" in result.lower() or "nonexistent" in result.lower()


# ── delete_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_task_found_confirms_deletion():
    from aug.core.tools.tasks import delete_task

    conn = _make_conn(fetchrow=_task_row(), fetchval="uuid-1")

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await delete_task.ainvoke({"name": "morning-news"}, config=_DM_CONFIG)

    assert "deleted" in result.lower() or "morning-news" in result


@pytest.mark.asyncio
async def test_delete_task_not_found_returns_not_found_message():
    from aug.core.tools.tasks import delete_task

    conn = _make_conn(fetchrow=None)

    with patch("aug.utils.db._pool", _make_pool(conn)):
        result = await delete_task.ainvoke({"name": "nonexistent"}, config=_DM_CONFIG)

    assert "not found" in result.lower() or "nonexistent" in result.lower()
