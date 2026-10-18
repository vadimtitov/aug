"""MCPManager — connects to configured MCP servers at startup and exposes their
tools as native LangChain tools. Only stdio (npx/uvx) and streamable-HTTP
transports are supported; a bad server config is isolated (logged + skipped),
never blocking the others or AUG's own startup.

Each server owns exactly one task for its whole connected lifetime: AnyIO task
groups (opened internally by stdio/streamable-HTTP sessions) require the task
that enters a cancel scope to be the one that exits it, so only that owner
task ever touches the session's context manager directly.
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
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import anyio
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.stdio import get_default_environment
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

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
# The un-namespaced half of a tool name, before "{server}__" is prepended —
# function-calling APIs restrict names to this set, so a server whose tool
# name doesn't qualify must be skipped rather than handed to the LLM broken.
_TOOL_NAME_CHARS_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_ERROR_LEN = 300
# Transport-death signals a tool call can observe directly — a broken pipe or
# a session someone already closed. conn.stop.wait() never notices these on
# its own (nothing signals it), so a tool call catching one is the only place
# health finds out the server actually died, rather than staying "active"
# until the next restart.
_TRANSPORT_DEAD_EXCEPTIONS = (anyio.ClosedResourceError, anyio.BrokenResourceError)


def _is_connection_closed(exc: BaseException) -> bool:
    """True for an ``McpError`` raised because the read stream closed (e.g. the
    server process exited) — a second transport-death check alongside
    ``_TRANSPORT_DEAD_EXCEPTIONS``."""
    return isinstance(exc, McpError) and exc.error.code == CONNECTION_CLOSED


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
        """Close every open MCP session. Call once during app shutdown — each
        connection is closed by signalling and awaiting its own owner task."""
        connections = list(self._connections.values())
        self._connections.clear()
        await asyncio.gather(
            *(self._close_one(conn) for conn in connections), return_exceptions=True
        )

    async def reconcile_operations(self) -> list[McpOperationOutcome]:
        """Resolve any install/remove operation a previous boot left unresolved
        (``saved`` or ``restart_pending`` — either way the config change is
        already on disk and was attempted this boot). Returns one outcome per
        resolved operation, for the caller to deliver back to whoever requested it.
        """
        # Cheap, lock-free fast path: the overwhelmingly common case (no
        # install/remove ever ran) must not perform a write on every single
        # boot. A concurrent writer racing this uncontended peek is not a
        # real risk here — this only ever runs once, at startup.
        if not any(op.state in ("saved", "restart_pending") for op in load_state().mcp.operations):
            return []

        configured_names = {s.name for s in load_settings().mcp_servers}

        outcomes = []
        async with update_state() as state:
            pending = [
                op for op in state.mcp.operations if op.state in ("saved", "restart_pending")
            ]
            for op in pending:
                health = self.health.get(op.server_name)
                if op.action == "remove":
                    # Both must be true: gone from config (nobody will ever
                    # reconnect it) and not an active session (nothing is
                    # actually still serving tools) — a server that's merely
                    # disabled or mid-failure but still listed in settings.json
                    # is not a completed removal.
                    still_configured = op.server_name in configured_names
                    still_active = health is not None and health.status == "active"
                    succeeded = not still_configured and not still_active
                    failure_detail = (
                        "server is still in configuration"
                        if still_configured
                        else "server is still connected after restart"
                    )
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

        # Everything below is cancellable — load_all()'s overall deadline can
        # cancel this coroutine at any await point, including mid-startup.
        # CancelledError is a BaseException, so it skips straight past the
        # `except TimeoutError`/`except Exception` blocks below without the
        # `finally` here, leaving conn.task registered nowhere: not in
        # self._connections (only a successful finish adds it) and no longer
        # reachable to cancel, so aclose() can never find and close it. The
        # `finally` runs on the normal-return paths too — see `registered`.
        registered = False
        try:
            try:
                await asyncio.wait_for(conn.ready.wait(), timeout=_PER_SERVER_TIMEOUT)
            except TimeoutError:
                conn.error = conn.error or TimeoutError(
                    f"connection timed out after {_PER_SERVER_TIMEOUT:.0f}s"
                )

            if conn.error is not None:
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
                logger.warning(
                    "mcp server %s: failed to load tools: %s", cfg.name, _sanitize_error(exc)
                )
                self.health[cfg.name] = McpServerHealth(
                    cfg.name, cfg.transport, "failed", error=_sanitize_error(exc)
                )
                return

            self._connections[cfg.name] = conn
            registered = True
            self.tools.extend(tools)
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "active", tool_count=len(tools)
            )
            logger.info("mcp server %s: connected, %d tool(s)", cfg.name, len(tools))
        finally:
            if not registered:
                await self._abandon(conn)

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
            if not _TOOL_NAME_CHARS_RE.match(t.name):
                logger.warning(
                    "mcp server %s: tool name %r has unsupported characters, skipped",
                    cfg.name,
                    t.name,
                )
                continue
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
            namespaced.append(
                _namespace_tool(
                    t,
                    new_name,
                    on_transport_death=lambda exc, name=cfg.name: self._mark_transport_dead(
                        name, exc
                    ),
                )
            )
        return namespaced

    def _mark_transport_dead(self, server_name: str, exc: BaseException) -> None:
        """A tool call just observed the transport is gone — mark health failed
        and wake the owner task so it unwinds the broken session."""
        prior = self.health.get(server_name)
        transport = prior.transport if prior else ""
        logger.warning(
            "mcp server %s: transport lost during tool call: %s", server_name, _sanitize_error(exc)
        )
        self.health[server_name] = McpServerHealth(
            server_name, transport, "failed", error=_sanitize_error(exc)
        )
        conn = self._connections.get(server_name)
        if conn is not None:
            conn.stop.set()


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


def _namespace_tool(
    t: BaseTool,
    new_name: str,
    *,
    on_transport_death: Callable[[BaseException], None] | None = None,
) -> BaseTool:
    """Rename an MCP-derived tool, bound to the per-call timeout, and turn expected
    failures (timeout, transport errors) into a ToolException instead of letting
    them escape the graph. ``on_transport_death``, when given, fires on a dead
    connection so MCPManager can mark the server's health failed. Every returned
    string is redacted — a resolved secret can echo back in a success result too.
    """
    original_coroutine = t.coroutine

    async def _timed(*args, **kwargs):
        try:
            result = await asyncio.wait_for(
                original_coroutine(*args, **kwargs), timeout=_TOOL_CALL_TIMEOUT
            )
        except TimeoutError as exc:
            raise ToolException(
                f"Tool '{new_name}' did NOT complete: timed out after {_TOOL_CALL_TIMEOUT:.0f}s."
            ) from exc
        except _TRANSPORT_DEAD_EXCEPTIONS as exc:
            if on_transport_death is not None:
                on_transport_death(exc)
            raise ToolException(
                f"Tool '{new_name}' did NOT complete: server connection lost "
                f"({_sanitize_error(exc)})."
            ) from exc
        except ToolException as exc:
            raise ToolException(_redact(str(exc))) from exc
        except Exception as exc:
            # A dead child process (e.g. an MCP server calling os._exit) is not
            # always observed as a raw anyio stream error — the session's own
            # receive loop can resolve the in-flight request with an McpError
            # carrying CONNECTION_CLOSED instead (see mcp.shared.session._receive_loop).
            # That must be recognized as transport death too, or health stays
            # "active" until the next restart.
            if on_transport_death is not None and _is_connection_closed(exc):
                on_transport_death(exc)
            raise ToolException(
                f"Tool '{new_name}' did NOT complete: {_sanitize_error(exc)}"
            ) from exc
        return _redact_result(result)

    return StructuredTool(
        name=new_name,
        description=t.description,
        args_schema=t.args_schema,
        coroutine=_timed,
        response_format=t.response_format,
        metadata=t.metadata,
        handle_tool_error=True,
    )


def _redact_result(result):
    """Scrub a successful tool result the same way an error message is scrubbed.
    MCP results nest content blocks inside a list inside a (content, artifact)
    tuple, so this walks the whole structure rather than assuming a bare string.
    """
    if isinstance(result, str):
        return _redact(result)
    if isinstance(result, dict):
        return {k: _redact_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_redact_result(v) for v in result]
    if isinstance(result, tuple):
        return tuple(_redact_result(v) for v in result)
    return result


# Every plaintext secret value this process has ever resolved via `hushed`,
# for any MCP server — see _resolve_hushed_ref. `_redact`/`_sanitize_error`
# scrub against this so a leaked value can never reach a log line, a tool
# result, or a health/error string regardless of which code path produced it.
_known_secrets: set[str] = set()


def _register_secret(value: str) -> None:
    if value:
        _known_secrets.add(value)


def _redact(text: str) -> str:
    for secret in _known_secrets:
        if secret in text:
            text = text.replace(secret, "[REDACTED]")
    return text


def _sanitize_error(exc: BaseException) -> str:
    """A short, safe-to-log-or-return summary of *exc* — redacted and length-capped.
    Deliberately ``str(exc)``, never repr: a transport exception's repr can carry
    a whole request (headers, URL, body), exactly where a secret would show up.
    """
    text = str(exc) or exc.__class__.__name__
    return _redact(text)[:_MAX_ERROR_LEN]


class _StderrPump:
    """Captures a stdio MCP server's stderr and forwards redacted lines to our own
    logger. ``stdio_client``'s ``errlog`` reaches the child as a raw fd via
    ``dup2``, so redaction has to happen on our end of the pipe, not by wrapping
    ``sys.stderr``.
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
        # Inherited defaults (PATH, HOME, ...) first, then the registry's
        # static (literal, non-secret) values on top, then resolved hushed
        # secrets last — so an explicitly bound credential always wins over
        # both the inherited environment and a registry-declared default.
        resolved_env = await _resolve_stdio_env(cfg.env_static, cfg.env)
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
            "headers": {**cfg.headers_static, **await _resolve_refs(cfg.headers)},
        }
        async with create_session(connection) as session:
            yield session


async def _resolve_stdio_env(static: dict[str, str], declared: dict[str, str]) -> dict[str, str]:
    """Scrubbed subprocess env: MCP's own safe defaults (PATH, HOME, ...), then
    registry literal defaults, then hushed-resolved secrets — later wins. Never
    AUG's own process environment (API_KEY, DATABASE_URL, ...)."""
    env = get_default_environment()
    env.update(static)
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
    value = await asyncio.to_thread(_read_hushed_secret, key)
    _register_secret(value)
    return value


def _read_hushed_secret(name: str) -> str:
    """Fetch one secret's value from hushed via a private inherited pipe, since
    `hushed run` redacts secret values from stdout. The reader is a tiny Python
    child (fd passed as a plain argv int, not shell ``>&N`` redirection, which
    breaks past descriptor 9 on Debian's dash).
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
