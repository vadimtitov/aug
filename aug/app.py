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
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aug.api.interfaces.telegram import TelegramInterface
from aug.api.routers import chat, files, gmail_auth, threads
from aug.config import get_settings
from aug.core.memory import init_memory_files, start_consolidation_scheduler
from aug.utils.db import create_pool
from aug.utils.logging import configure_logging, set_correlation_id
from aug.utils.reminders import start_reminder_loop
from aug.utils.storage import LocalFileStorage

logger = logging.getLogger(__name__)

_BANNER = r"""
   █████╗ ██╗   ██╗ ██████╗
  ██╔══██╗██║   ██║██╔════╝
  ███████║██║   ██║██║  ███╗
  ██╔══██║██║   ██║██║   ██║
  ██║  ██║╚██████╔╝╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝

"""


@asynccontextmanager
async def _checkpointer_context(dsn: str):
    """Async context manager that owns the Postgres checkpointer lifetime.

    Import is deferred so the module loads without libpq (tests mock this).
    ``from_conn_string`` returns a context manager in langgraph-checkpoint-postgres 2+.
    """

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("aug.core.tools.approval", "ApprovalRequest"),
            ("aug.core.tools.approval", "ApprovalDecision"),
        ]
    )
    async with AsyncPostgresSaver.from_conn_string(dsn, serde=serde) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of shared resources."""
    configure_logging(debug=get_settings().DEBUG)
    logger.info("AUG %s starting up…", get_settings().APP_VERSION)
    init_memory_files()

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

        # Interface registry — keyed by interface name, used for proactive notifications
        app.state.interfaces = {}
        app.state.last_reminder_check = None

        telegram = TelegramInterface(checkpointer)
        await telegram.start_polling(app)

        consolidation_task = await start_consolidation_scheduler()
        reminder_task = start_reminder_loop(app)

        sys.stdout.flush()
        sys.stdout.write(_BANNER)
        sys.stdout.flush()
        settings = get_settings()
        logger.info(
            "AUG startup complete — version=%s telegram=%s brave=%s gmail=%s portainer=%s",
            settings.APP_VERSION,
            bool(settings.TELEGRAM_BOT_TOKEN),
            bool(settings.BRAVE_API_KEY),
            bool(settings.GMAIL_CLIENT_ID),
            bool(settings.PORTAINER_URL),
        )
        yield

        consolidation_task.cancel()
        reminder_task.cancel()
        await telegram.stop_polling(app)

    # Shutdown
    await pool.close()
    logger.info("AUG shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AUG — Agent Using Graph",
        version=get_settings().APP_VERSION,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next) -> Response:
        cid = request.headers.get("X-Correlation-ID", str(uuid4())[:8])
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.get("/health", tags=["health"])
    async def health():
        checks: dict[str, object] = {"version": get_settings().APP_VERSION}

        # DB probe
        try:
            async with app.state.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["db"] = "ok"
        except Exception as exc:
            checks["db"] = f"error: {exc}"

        # Reminder loop watchdog
        last_check: datetime | None = app.state.last_reminder_check
        if last_check is None:
            checks["reminder_loop"] = "not_started"
        elif datetime.now(tz=UTC) - last_check > timedelta(minutes=5):
            checks["reminder_loop"] = "stale"
        else:
            checks["reminder_loop"] = "ok"

        ok = checks["db"] == "ok" and checks["reminder_loop"] in ("ok", "not_started")
        checks["status"] = "ok" if ok else "degraded"
        return checks

    app.include_router(chat.router)
    app.include_router(threads.router)
    app.include_router(files.router)
    app.include_router(gmail_auth.router)

    return app


app = create_app()
