"""Postgres connection pool via asyncpg.

The pool is created once at application startup and stored on ``app.state``.
All routers access it via ``request.app.state.db_pool``.

Schema bootstrap:
    On first startup ``_ensure_schema()`` creates the ``threads`` table if it
    does not already exist.  In production you would replace this with proper
    migrations (Alembic, Flyway, etc.).
"""

import asyncio
import logging
import re

import asyncpg

logger = logging.getLogger(__name__)

_CREATE_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id    TEXT PRIMARY KEY,
    agent_version TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);
"""

_CREATE_REMINDERS_TABLE = """
CREATE TABLE IF NOT EXISTS reminders (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message           TEXT NOT NULL,
    trigger_at        TIMESTAMPTZ NOT NULL,
    fired             BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notification_interface  TEXT NOT NULL DEFAULT '',
    notification_target     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS reminders_pending_idx
    ON reminders (trigger_at) WHERE fired = FALSE;
"""

_CREATE_SCHEDULED_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    interface       TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    message         TEXT NOT NULL,
    schedule_type   TEXT NOT NULL,
    schedule_params JSONB NOT NULL DEFAULT '{}',
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    push_type       TEXT NOT NULL DEFAULT 'agent',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Idempotent migrations — add columns introduced after initial schema creation.
_MIGRATE_REMINDERS_COLUMNS = """
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS notification_interface TEXT NOT NULL DEFAULT '';
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS notification_target     TEXT NOT NULL DEFAULT '';
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS retry_count      INTEGER     NOT NULL DEFAULT 0;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS next_retry_at    TIMESTAMPTZ;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS last_error       TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS dead_lettered_at TIMESTAMPTZ;
"""

_MIGRATE_SCHEDULED_TASKS_COLUMNS = """
ALTER TABLE scheduled_tasks ADD COLUMN IF NOT EXISTS push_type TEXT NOT NULL DEFAULT 'agent';
"""

# One-time migration: move unfired future reminders into scheduled_tasks so they
# are delivered by the APScheduler path after the reminder loop is removed.
_MIGRATE_REMINDERS_TO_TASKS = """
INSERT INTO scheduled_tasks
    (name, interface, thread_id, message, schedule_type, schedule_params, push_type, created_at)
SELECT
    'reminder-' || substr(id::text, 1, 8) AS name,
    notification_interface                 AS interface,
    CASE
        WHEN notification_interface = 'telegram' AND notification_target <> ''
            THEN 'default:' || notification_target
        ELSE 'default'
    END                                    AS thread_id,
    chr(9200) || ' ' || message            AS message,
    'date'                                 AS schedule_type,
    jsonb_build_object('run_date', trigger_at::text) AS schedule_params,
    'forward'                              AS push_type,
    created_at
FROM reminders
WHERE fired = FALSE
  AND dead_lettered_at IS NULL
  AND trigger_at > NOW()
ON CONFLICT (name) DO NOTHING;
"""


_pool: asyncpg.Pool | None = None


def set_pool(pool: asyncpg.Pool) -> None:
    """Store the application pool so tools and background jobs can use it."""
    global _pool
    _pool = pool


def get_pool() -> asyncpg.Pool:
    """Return the application pool.  Raises RuntimeError if not yet initialized."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call set_pool() at startup")
    return _pool


def strip_driver(url: str) -> str:
    """Convert ``postgresql+asyncpg://...`` → ``postgresql://...`` for asyncpg."""
    return re.sub(r"^postgresql\+asyncpg://", "postgresql://", url)


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Waits up to 90 seconds for Postgres to become ready before giving up.

    Args:
        database_url: Full connection URI, e.g.
            ``postgresql+asyncpg://user:password@host:5432/dbname``.
    """
    dsn = strip_driver(database_url)
    await _wait_for_postgres(dsn)
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=15,
        command_timeout=30,
        server_settings={"application_name": "aug"},
    )
    await _ensure_schema(pool)
    logger.info("Postgres ready — %s", dsn.split("@")[-1])
    return pool


async def _wait_for_postgres(dsn: str) -> None:
    """Probe Postgres with exponential back-off until it accepts connections.

    Gives up after 90 seconds and raises TimeoutError.
    """
    delay = 1.0
    try:
        async with asyncio.timeout(90):
            while True:
                try:
                    conn = await asyncpg.connect(dsn=dsn)
                    await conn.close()
                    return
                except Exception as exc:
                    logger.warning("Postgres not ready — retrying in %.0fs (%s)", delay, exc)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
    except TimeoutError:
        raise RuntimeError("Postgres did not become ready within 90s") from None


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    """Create application tables if they don't exist yet."""
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_THREADS_TABLE)
        await conn.execute(_CREATE_REMINDERS_TABLE)
        await conn.execute(_MIGRATE_REMINDERS_COLUMNS)
        await conn.execute(_CREATE_SCHEDULED_TASKS_TABLE)
        await conn.execute(_MIGRATE_SCHEDULED_TASKS_COLUMNS)
        await conn.execute(_MIGRATE_REMINDERS_TO_TASKS)
    logger.debug("DB schema verified.")
