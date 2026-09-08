"""FastAPI application factory.

Startup sequence:
1. Configure structured logging.
2. Open Postgres connection pool.
3. Create LangGraph Postgres checkpointer (shared across all agents).
4. Mount API routers.
5. Optionally start Telegram polling bot.
6. Announce the boot on every interface that has a push channel.

All shared resources are stored on ``app.state`` so routers can access them
via ``request.app.state.<resource>``.
"""

import asyncio
import logging
import re
import sys
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aug.api.interfaces.telegram import TelegramInterface
from aug.api.internal.gateway import serve_gateway
from aug.api.routers import (
    auth,
    browser,
    chat,
    files,
    gmail_auth,
    hooks,
    oauth,
    settings,
    skills,
    threads,
)
from aug.config import get_settings
from aug.core.browser_view import BrowserViewHub
from aug.core.dispatch import broadcast
from aug.core.dispatch import set_app as set_push_app
from aug.core.memory import init_memory_files, start_consolidation_scheduler
from aug.core.oauth.providers import PROVIDERS_FILE, ProviderRegistry
from aug.core.skill_deps import warm_all_skills
from aug.utils.db import create_pool, set_pool
from aug.utils.logging import configure_logging, set_correlation_id
from aug.utils.ratelimit import RateLimiter
from aug.utils.scheduler import start_scheduler, stop_scheduler
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


async def _announce_startup(app: FastAPI) -> None:
    """Tell every interface with a push channel that AUG is back up.

    Runs as a background task: broadcast is best-effort and swallows its own
    delivery failures, so an unreachable chat can neither delay nor fail the boot.
    """
    if not get_settings().STARTUP_ANNOUNCEMENT:
        return
    delivered = await broadcast(app, f"🟢 AUG {get_settings().APP_VERSION} is up.")
    logger.info("startup announcement delivered to %d thread(s)", delivered)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of shared resources."""
    configure_logging(debug=get_settings().DEBUG)
    logger.info("AUG %s starting up…", get_settings().APP_VERSION)
    init_memory_files()

    # Database pool
    pool = await create_pool(get_settings().DATABASE_URL)
    app.state.db_pool = pool
    set_pool(pool)

    # LangGraph Postgres checkpointer
    # Strip the +asyncpg driver suffix — psycopg expects a plain postgres:// URI.
    dsn = re.sub(r"^postgresql\+asyncpg://", "postgresql://", get_settings().DATABASE_URL)
    async with _checkpointer_context(dsn) as checkpointer:
        app.state.checkpointer = checkpointer

        # File storage
        app.state.storage = LocalFileStorage()

        # OAuth — provider registry, and the transport the flow uses (None = default).
        app.state.oauth_providers = ProviderRegistry(PROVIDERS_FILE)
        app.state.oauth_transport = None
        # Both public OAuth endpoints are unauthenticated by necessity — see the
        # design doc.  Per client IP, refused before any database work happens.
        app.state.oauth_start_limiter = RateLimiter(limit=30, per_seconds=3600)
        app.state.oauth_callback_limiter = RateLimiter(limit=10, per_seconds=60)

        # Interface registry — keyed by interface name, used for proactive notifications
        app.state.interfaces = {}

        # Live browser view — lazy; the screencast only runs while someone watches.
        app.state.browser_view_hub = BrowserViewHub(get_settings().BROWSER_CDP_URL)

        telegram = TelegramInterface(checkpointer)
        await telegram.start_polling(app)

        set_push_app(app)
        consolidation_task = await start_consolidation_scheduler()
        scheduler_task = await start_scheduler(app)

        # Pre-resolve installed skills' PEP 723 dependencies so the first agent run after
        # a rebuild doesn't stall on downloads. Background + best-effort: never blocks
        # startup, runs off the event loop (uv shells out, which is blocking).
        warmup_task = asyncio.create_task(asyncio.to_thread(warm_all_skills))

        # Loopback token gateway — lets the agent use OAuth credentials it cannot read.
        gateway_task = asyncio.create_task(serve_gateway(app.state))

        announce_task = asyncio.create_task(_announce_startup(app))

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

        gateway_task.cancel()
        announce_task.cancel()
        consolidation_task.cancel()
        scheduler_task.cancel()
        warmup_task.cancel()
        await stop_scheduler(app)
        await telegram.stop_polling(app)
        await app.state.browser_view_hub.aclose()

    # Shutdown
    await pool.close()
    logger.info("AUG shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AUG — Agent Using Graph",
        version=get_settings().APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "PUT", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
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
        except Exception:
            checks["db"] = "error"

        checks["status"] = "ok" if checks["db"] == "ok" else "degraded"
        return checks

    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(threads.router)
    app.include_router(files.router)
    app.include_router(gmail_auth.router)
    app.include_router(settings.router)
    app.include_router(skills.router)
    app.include_router(hooks.router)
    app.include_router(browser.router)
    app.include_router(oauth.router)

    return app


app = create_app()
