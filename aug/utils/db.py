"""Postgres connection pool via asyncpg.

The pool is created once at application startup and stored on ``app.state``.
All routers access it via ``request.app.state.db_pool``.

Schema bootstrap:
    On first startup ``_ensure_schema()`` creates the ``threads`` table if it
    does not already exist.  In production you would replace this with proper
    migrations (Alembic, Flyway, etc.).
"""

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

# Idempotent migrations — add columns introduced after initial schema creation.
_MIGRATE_REMINDERS_COLUMNS = """
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS notification_interface TEXT NOT NULL DEFAULT '';
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS notification_target     TEXT NOT NULL DEFAULT '';
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS retry_count   INTEGER   NOT NULL DEFAULT 0;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS last_error    TEXT;
"""


def _strip_driver(url: str) -> str:
    """Convert ``postgresql+asyncpg://...`` → ``postgresql://...`` for asyncpg."""
    return re.sub(r"^postgresql\+asyncpg://", "postgresql://", url)


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Args:
        database_url: Full connection URI, e.g.
            ``postgresql+asyncpg://user:password@host:5432/dbname``.
    """
    dsn = _strip_driver(database_url)
    pool: asyncpg.Pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
    logger.info("Postgres pool created — %s", dsn.split("@")[-1])
    await _ensure_schema(pool)
    return pool


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    """Create application tables if they don't exist yet."""
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_THREADS_TABLE)
        await conn.execute(_CREATE_REMINDERS_TABLE)
        await conn.execute(_MIGRATE_REMINDERS_COLUMNS)
    logger.debug("DB schema verified.")
