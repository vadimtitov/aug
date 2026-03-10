"""FastAPI application factory.

Startup sequence:
1. Configure structured logging.
2. Open Postgres connection pool.
3. Create LangGraph Postgres checkpointer (shared across all agents).
4. Mount API routers.
5. Optionally start Telegram polling bot.

All shared resources are stored on ``app.state`` so routers can access them
via ``request.app.state.<resource>``.
"""

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aug.api.routers import chat, files, threads
from aug.config import get_settings
from aug.utils.db import create_pool
from aug.utils.logging import configure_logging
from aug.utils.storage import LocalFileStorage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _checkpointer_context(dsn: str):
    """Async context manager that owns the Postgres checkpointer lifetime.

    Import is deferred so the module loads without libpq (tests mock this).
    ``from_conn_string`` returns a context manager in langgraph-checkpoint-postgres 2+.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of shared resources."""
    configure_logging(debug=get_settings().DEBUG)
    logger.info("AUG %s starting up…", get_settings().APP_VERSION)

    # Database pool
    pool = await create_pool(get_settings().DATABASE_URL)
    app.state.db_pool = pool

    # LangGraph Postgres checkpointer
    # Strip the +asyncpg driver suffix — psycopg expects a plain postgres:// URI.
    dsn = re.sub(r"^postgresql\+asyncpg://", "postgresql://", get_settings().DATABASE_URL)
    async with _checkpointer_context(dsn) as checkpointer:
        app.state.checkpointer = checkpointer

        # File storage
        app.state.storage = LocalFileStorage()

        # Optional Telegram bot (polling)
        from aug.api.telegram import start_polling, stop_polling

        await start_polling(app)

        logger.info("AUG startup complete.")
        yield

        await stop_polling(app)

    # Shutdown
    await pool.close()
    logger.info("AUG shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AUG — Agent Using Graph",
        version=get_settings().APP_VERSION,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": get_settings().APP_VERSION}

    app.include_router(chat.router)
    app.include_router(threads.router)
    app.include_router(files.router)

    return app


app = create_app()
