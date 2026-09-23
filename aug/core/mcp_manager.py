"""MCPManager — connects to configured MCP servers at startup and exposes their
tools as native LangChain tools.

Startup flow (see ``aug/app.py`` ``lifespan()``)::

    manager = MCPManager()
    await manager.load_all(base_tool_names=...)
    configure_mcp_tools(manager.tools)   # aug/core/registry.py
    set_manager(manager)                 # so tools/mcp.py can read health/trigger restarts

Only stdio (npx/uvx) and streamable-HTTP transports are supported — Docker-based
MCP servers are out of scope for v1 (no Docker socket access; see the PRD).

Failures are isolated per server: one bad config logs a warning and is skipped,
it never blocks the others or aborts AUG's own startup. If every server fails,
AUG boots normally with just its base tools.

Each server owns exactly one task for its whole connected lifetime (see
``_ServerConnection`` / ``MCPManager._run_session``): that task is the only one
that ever enters or exits the session's context manager, because AnyIO task
groups (which stdio/streamable-HTTP sessions open internally) require the task
that enters a cancel scope to be the one that exits it. Everything else —
``load_all``, tool calls, ``aclose`` — talks to the session by reference or by
signalling the owner task, never by touching its context manager directly.
"""

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.stdio import get_default_environment

from aug.utils.file_settings import McpServerConfig, load_settings
from aug.utils.state import McpOperation, load_state, update_state

logger = logging.getLogger(__name__)

_PER_SERVER_TIMEOUT = 30.0
_OVERALL_TIMEOUT = 60.0
_SHUTDOWN_TIMEOUT = 10.0
_MAX_CONCURRENT_CONNECTS = 4
_TOOL_CALL_TIMEOUT = 60.0
_HUSHED_TIMEOUT = 15.0
_HUSHED_PREFIX = "hushed:"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# OpenAI/Anthropic function-calling name limits top out around 64 characters —
# a single over-long MCP tool name must not prevent every *other* tool in the
# request from binding.
_MAX_TOOL_NAME_LEN = 64
_MAX_ERROR_LEN = 300

# A tiny, argv-driven reader: `hushed run` injects the secret into this child's
# environment under NAME, and it writes the raw bytes straight to the fd number
# given on argv. Passing the fd as an int argument (not shell syntax) is what
# lets this work past descriptor 9 — see _read_hushed_secret.
_HUSHED_READER_SCRIPT = (
    "import os,sys\n"
    "fd = int(sys.argv[1])\n"
    "os.write(fd, os.environb.get(sys.argv[2].encode(), b''))\n"
)


class McpSecretError(Exception):
    """A declared ``hushed:KEY`` reference could not be resolved."""


@dataclass
class McpServerHealth:
    name: str
    transport: str
    status: str  # "active" | "failed"
    tool_count: int = 0
    error: str = ""


@dataclass
class McpOperationOutcome:
    """One resolved install/remove operation, ready to report.

    ``interface``/``thread_id`` are empty for operations recorded before this
    field existed — callers should fall back to a general broadcast for those.
    """

    server_name: str
    summary: str
    interface: str = ""
    thread_id: str = ""


@dataclass
class _ServerConnection:
    """State shared between MCPManager and one server's owner task."""

    name: str
    task: asyncio.Task | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    session: ClientSession | None = None
    error: BaseException | None = None


class MCPManager:
    """Owns every live MCP session for the container's lifetime."""

    def __init__(self) -> None:
        self.tools: list[BaseTool] = []
        self.health: dict[str, McpServerHealth] = {}
        self._tool_names: set[str] = set()
        self._connections: dict[str, _ServerConnection] = {}

    async def load_all(self, *, base_tool_names: set[str] | None = None) -> None:
        """Connect to every enabled server in settings.json, bounded by an overall
        deadline. ``base_tool_names`` seeds collision detection against the
        agent's non-MCP tools."""
        self._tool_names = set(base_tool_names or set())
        servers = [s for s in load_settings().mcp_servers if s.enabled]
        if not servers:
            return

        sem = asyncio.Semaphore(_MAX_CONCURRENT_CONNECTS)

        async def _bounded(cfg: McpServerConfig) -> None:
            async with sem:
                await self._load_one(cfg)

        try:
            await asyncio.wait_for(
                asyncio.gather(*(_bounded(cfg) for cfg in servers)),
                timeout=_OVERALL_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("mcp_manager: startup deadline (%.0fs) exceeded", _OVERALL_TIMEOUT)

    async def aclose(self) -> None:
        """Close every open MCP session. Call once during app shutdown.

        Each connection is closed by signalling and awaiting its own owner
        task — never by reaching into its session's context manager from this
        (different) task.
        """
        connections = list(self._connections.values())
        self._connections.clear()
        await asyncio.gather(
            *(self._close_one(conn) for conn in connections), return_exceptions=True
        )

    async def reconcile_operations(self) -> list[McpOperationOutcome]:
        """Resolve any install/remove operation a previous boot left unresolved
        — at ``restart_pending`` because the restart it triggered crashed AUG
        before it could record the outcome, or even still at ``saved`` because
        AUG died (for any reason) in the few seconds between saving config and
        triggering that restart, before ``_trigger_restart`` ever got to mark
        it pending. Either way the config is already on disk and will have been
        attempted this boot, so both states are resolved the same way here.
        Returns one outcome per resolved operation, for the caller to deliver
        back to whoever requested it (and to fold into the general startup
        announcement as a fallback).
        """
        # Cheap, lock-free fast path: the overwhelmingly common case (no
        # install/remove ever ran) must not perform a write on every single
        # boot. A concurrent writer racing this uncontended peek is not a
        # real risk here — this only ever runs once, at startup.
        if not any(op.state in ("saved", "restart_pending") for op in load_state().mcp.operations):
            return []

        outcomes = []
        async with update_state() as state:
            pending = [
                op for op in state.mcp.operations if op.state in ("saved", "restart_pending")
            ]
            for op in pending:
                health = self.health.get(op.server_name)
                if op.action == "remove":
                    succeeded = health is None or health.status != "active"
                    failure_detail = "server is still connected after restart"
                else:
                    succeeded = health is not None and health.status == "active"
                    failure_detail = health.error if health else "server not found after restart"

                op.state = "active" if succeeded else "failed"
                op.detail = "" if succeeded else failure_detail
                tool_count = health.tool_count if (succeeded and health) else 0
                outcomes.append(
                    McpOperationOutcome(
                        server_name=op.server_name,
                        summary=_operation_summary(op, succeeded, tool_count),
                        interface=op.interface,
                        thread_id=op.thread_id,
                    )
                )
        return outcomes

    async def _load_one(self, cfg: McpServerConfig) -> None:
        conn = _ServerConnection(name=cfg.name)
        conn.task = asyncio.create_task(
            self._run_session(cfg, conn), name=f"mcp-session-{cfg.name}"
        )

        try:
            await asyncio.wait_for(conn.ready.wait(), timeout=_PER_SERVER_TIMEOUT)
        except TimeoutError:
            conn.error = conn.error or TimeoutError(
                f"connection timed out after {_PER_SERVER_TIMEOUT:.0f}s"
            )

        if conn.error is not None:
            await self._abandon(conn)
            logger.warning("mcp server %s: %s", cfg.name, _sanitize_error(conn.error))
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "failed", error=_sanitize_error(conn.error)
            )
            return

        try:
            tools = await asyncio.wait_for(
                self._namespaced_tools(cfg, conn.session), timeout=_PER_SERVER_TIMEOUT
            )
        except Exception as exc:
            await self._abandon(conn)
            logger.warning(
                "mcp server %s: failed to load tools: %s", cfg.name, _sanitize_error(exc)
            )
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "failed", error=_sanitize_error(exc)
            )
            return

        self._connections[cfg.name] = conn
        self.tools.extend(tools)
        self.health[cfg.name] = McpServerHealth(
            cfg.name, cfg.transport, "active", tool_count=len(tools)
        )
        logger.info("mcp server %s: connected, %d tool(s)", cfg.name, len(tools))

    async def _run_session(self, cfg: McpServerConfig, conn: _ServerConnection) -> None:
        """Owner task: the only task that ever enters or exits this server's
        session context, from connect through to the shutdown signal."""
        try:
            async with _open_session(cfg) as session:
                await asyncio.wait_for(session.initialize(), timeout=_PER_SERVER_TIMEOUT)
                conn.session = session
                conn.ready.set()
                await conn.stop.wait()
        except Exception as exc:
            was_live = conn.session is not None
            conn.error = exc
            conn.ready.set()
            if was_live and not conn.stop.is_set():
                # Not a startup failure (_load_one already returned) — the
                # session died mid-flight, so this is the only place able to
                # tell health about it.
                logger.warning(
                    "mcp server %s: session ended unexpectedly: %s", cfg.name, _sanitize_error(exc)
                )
                self.health[cfg.name] = McpServerHealth(
                    cfg.name, cfg.transport, "failed", error=_sanitize_error(exc)
                )

    async def _abandon(self, conn: _ServerConnection) -> None:
        """Unwind a connection that failed during setup, in its own owner
        task, immediately — a partially initialized session is never left for
        aclose() to discover later."""
        conn.task.cancel()
        with contextlib.suppress(BaseException):
            await conn.task

    async def _close_one(self, conn: _ServerConnection) -> None:
        conn.stop.set()
        try:
            await asyncio.wait_for(conn.task, timeout=_SHUTDOWN_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "mcp server %s: shutdown timed out after %.0fs, cancelling",
                conn.name,
                _SHUTDOWN_TIMEOUT,
            )
            conn.task.cancel()
            with contextlib.suppress(BaseException):
                await conn.task
        except Exception:
            pass  # _run_session already logged and recorded its own failure

    async def _namespaced_tools(
        self, cfg: McpServerConfig, session: ClientSession
    ) -> list[BaseTool]:
        raw_tools = await load_mcp_tools(session, server_name=cfg.name, tool_name_prefix=False)

        namespaced: list[BaseTool] = []
        for t in raw_tools:
            new_name = f"{cfg.name}__{t.name}"
            if len(new_name) > _MAX_TOOL_NAME_LEN:
                logger.warning(
                    "mcp server %s: tool name %r exceeds %d chars, skipped",
                    cfg.name,
                    new_name,
                    _MAX_TOOL_NAME_LEN,
                )
                continue
            if new_name in self._tool_names:
                logger.warning(
                    "mcp server %s: tool name collision on %r, skipped", cfg.name, new_name
                )
                continue
            self._tool_names.add(new_name)
            namespaced.append(_namespace_tool(t, new_name))
        return namespaced


# ---------------------------------------------------------------------------
# Module-level singleton — tools/mcp.py reads health / triggers restarts
# without needing app.state plumbed through every tool call, same pattern as
# aug/utils/db.py's get_pool()/set_pool().
# ---------------------------------------------------------------------------

_manager: MCPManager | None = None


def set_manager(manager: MCPManager | None) -> None:
    global _manager
    _manager = manager


def get_manager() -> MCPManager | None:
    """Return the live MCPManager, or None before startup has wired one in."""
    return _manager


# ---------------------------------------------------------------------------
# Durable operation state — install/remove tracked across the restart they
# trigger. See McpOperation / reconcile_operations().
# ---------------------------------------------------------------------------


async def record_operation(
    action: str, server_name: str, state: str, *, interface: str = "", thread_id: str = ""
) -> str:
    """Persist a new MCP operation record. Returns its id."""
    op_id = uuid.uuid4().hex[:8]
    async with update_state() as st:
        st.mcp.operations.append(
            McpOperation(
                id=op_id,
                action=action,
                server_name=server_name,
                state=state,
                created_at=time.time(),
                interface=interface,
                thread_id=thread_id,
            )
        )
    return op_id


async def update_operation_state(op_id: str, state: str, detail: str = "") -> None:
    async with update_state() as st:
        for op in st.mcp.operations:
            if op.id == op_id:
                op.state = state
                op.detail = detail
                break


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _operation_summary(op: McpOperation, succeeded: bool, tool_count: int) -> str:
    if succeeded:
        detail = f" ({tool_count} tools)" if op.action == "install" else ""
        return f"MCP server '{op.server_name}' {op.action} succeeded{detail}."
    return f"MCP server '{op.server_name}' {op.action} failed: {op.detail}"


def _namespace_tool(t: BaseTool, new_name: str) -> BaseTool:
    """Rename an MCP-derived tool, bound to the per-call timeout, and turn
    expected failures (timeout, transport/connection errors) into an explicit
    tool-error response instead of letting them escape the compiled graph —
    ToolException + handle_tool_error=True is what makes LangGraph's ToolNode
    emit a normal error ToolMessage rather than re-raising."""
    original_coroutine = t.coroutine

    async def _timed(*args, **kwargs):
        try:
            return await asyncio.wait_for(
                original_coroutine(*args, **kwargs), timeout=_TOOL_CALL_TIMEOUT
            )
        except TimeoutError as exc:
            raise ToolException(
                f"Tool '{new_name}' did NOT complete: timed out after {_TOOL_CALL_TIMEOUT:.0f}s."
            ) from exc
        except ToolException:
            raise
        except Exception as exc:
            raise ToolException(
                f"Tool '{new_name}' did NOT complete: {_sanitize_error(exc)}"
            ) from exc

    return StructuredTool(
        name=new_name,
        description=t.description,
        args_schema=t.args_schema,
        coroutine=_timed,
        response_format=t.response_format,
        metadata=t.metadata,
        handle_tool_error=True,
    )


def _sanitize_error(exc: BaseException) -> str:
    """A short, safe-to-log-or-return summary of *exc*.

    Deliberately ``str(exc)``, never ``%r``/``repr`` — a transport exception's
    repr can carry a whole request (headers, URL, body) verbatim, which is
    exactly where a resolved secret would show up. Capped in length so one
    verbose error (e.g. an HTML error page a broken proxy returned) can't
    flood logs or a tool result.
    """
    text = str(exc) or exc.__class__.__name__
    return text[:_MAX_ERROR_LEN]


class _StderrPump:
    """Captures a stdio MCP server's stderr and forwards redacted lines to
    our own logger, instead of letting the SDK pipe it straight to AUG's
    stderr fd.

    ``stdio_client``'s ``errlog`` only ever reaches ``subprocess.Popen`` as a
    raw file descriptor for OS-level ``dup2`` — the child writes directly to
    that descriptor, so nothing written there ever passes through a Python
    object's ``write()``, wrapping ``sys.stderr`` in one does nothing.
    Redaction has to happen on our own end of a pipe instead.
    """

    def __init__(self, server_name: str, secrets: Iterable[str]) -> None:
        self._server_name = server_name
        self._secrets = [s for s in secrets if s]
        self.read_fd, self.write_fd = os.pipe()
        self._task = asyncio.create_task(self._pump(), name=f"mcp-stderr-{server_name}")

    async def _pump(self) -> None:
        try:
            with os.fdopen(self.read_fd, "r", errors="replace") as f:
                while True:
                    line = await asyncio.to_thread(f.readline)
                    if not line:
                        return
                    for secret in self._secrets:
                        line = line.replace(secret, "[REDACTED]")
                    logger.info("mcp server %s stderr: %s", self._server_name, line.rstrip())
        except (OSError, ValueError):
            return

    async def aclose(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.write_fd)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(self._task, timeout=2.0)


@asynccontextmanager
async def _open_session(cfg: McpServerConfig) -> AsyncIterator[ClientSession]:
    """Open one server's session, entering and yielding it from a single
    ``async with`` — the caller (``MCPManager._run_session``) is what makes
    this the owner task for the session's whole lifetime."""
    if cfg.transport == "stdio":
        resolved_env = await _resolve_stdio_env(cfg.env)
        pump = _StderrPump(cfg.name, resolved_env.values())
        params = StdioServerParameters(command=cfg.command, args=cfg.args, env=resolved_env)
        try:
            async with (
                stdio_client(params, errlog=pump.write_fd) as (read, write),
                ClientSession(read, write) as session,
            ):
                yield session
        finally:
            await pump.aclose()
    else:
        connection = {
            "transport": "streamable_http",
            "url": cfg.url,
            "headers": await _resolve_refs(cfg.headers),
        }
        async with create_session(connection) as session:
            yield session


async def _resolve_stdio_env(declared: dict[str, str]) -> dict[str, str]:
    """Build a scrubbed subprocess env: the safe subset MCP itself would default
    to (PATH, HOME, ...) plus only the declared, hushed-resolved secrets.
    Never AUG's own process environment (API_KEY, DATABASE_URL, ...)."""
    env = get_default_environment()
    env.update(await _resolve_refs(declared))
    return env


async def _resolve_refs(declared: dict[str, str]) -> dict[str, str]:
    resolved = {}
    for key, ref in declared.items():
        resolved[key] = await _resolve_hushed_ref(ref)
    return resolved


async def _resolve_hushed_ref(ref: str) -> str:
    if not ref.startswith(_HUSHED_PREFIX):
        raise McpSecretError(
            "expected a 'hushed:KEY' reference but got a value in a different format "
            "(value withheld from this message — it may be an unintended plaintext secret)"
        )
    key = ref[len(_HUSHED_PREFIX) :]
    return await asyncio.to_thread(_read_hushed_secret, key)


def _read_hushed_secret(name: str) -> str:
    """Fetch one secret's value from hushed without it ever passing through this
    process's stdout/stderr — ``hushed run`` redacts secret values from the
    wrapped command's output, so printing it to stdout comes back as
    "[REDACTED]" rather than the value. Instead, a private pipe is inherited
    into the wrapped process, which writes the value there directly — a
    channel hushed never inspects — and we read it back once the (bounded,
    non-interactive) process has exited.

    The reader is a tiny Python child, not ``sh -c '... >&{fd}'``: the file
    descriptor is passed as a plain argv string and turned into an int, not
    shell redirection syntax, so it works for any descriptor number — `sh`'s
    ``>&N`` redirection only accepts single-digit N on Debian's dash, which
    silently broke every secret above descriptor 9 (AUG already has several
    open — DB pool, sockets — before MCP servers load).

    ``name`` is passed as an argv element, never interpolated into a command
    string, so it can't reach a shell unescaped regardless of where it came
    from (settings.json, ultimately sourced from an MCP registry listing).
    It's still validated up front as a normal env-var name — not for shell
    safety, but so a malformed name fails clearly instead of hushed silently
    not finding it.
    """
    if not _ENV_VAR_NAME_RE.match(name):
        raise McpSecretError(f"invalid secret name {name!r} — must be a valid env var name")

    with tempfile.TemporaryFile() as tf:
        fd = tf.fileno()
        try:
            result = subprocess.run(
                ["hushed", "run", "--", sys.executable, "-c", _HUSHED_READER_SCRIPT, str(fd), name],
                pass_fds=(fd,),
                capture_output=True,
                text=True,
                timeout=_HUSHED_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise McpSecretError(f"failed to read secret {name!r}: {exc}") from exc
        if result.returncode != 0:
            raise McpSecretError(
                f"hushed failed resolving {name!r}: {result.stderr.strip()[:_MAX_ERROR_LEN]}"
            )
        tf.seek(0)
        value = tf.read().decode("utf-8", errors="replace")

    if not value:
        raise McpSecretError(f"secret {name!r} is not set. Set it with: hushed add {name} <value>")
    return value
