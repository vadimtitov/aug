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
    logger.debug("DB schema verified.")
