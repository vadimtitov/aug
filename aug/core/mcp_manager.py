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
"""

import asyncio
import logging
import re
import subprocess
import tempfile
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

from aug.utils.file_settings import McpServerConfig, load_settings
from aug.utils.state import McpOperation, load_state, save_state

logger = logging.getLogger(__name__)

_PER_SERVER_TIMEOUT = 30.0
_OVERALL_TIMEOUT = 60.0
_MAX_CONCURRENT_CONNECTS = 4
_TOOL_CALL_TIMEOUT = 60.0
_HUSHED_TIMEOUT = 15.0
_HUSHED_PREFIX = "hushed:"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class McpSecretError(Exception):
    """A declared ``hushed:KEY`` reference could not be resolved."""


@dataclass
class McpServerHealth:
    name: str
    transport: str
    status: str  # "active" | "failed"
    tool_count: int = 0
    error: str = ""


class MCPManager:
    """Owns every live MCP session for the container's lifetime."""

    def __init__(self) -> None:
        self.tools: list[BaseTool] = []
        self.health: dict[str, McpServerHealth] = {}
        self._exit_stack = AsyncExitStack()
        self._tool_names: set[str] = set()

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
        """Close every open MCP session. Call once during app shutdown."""
        await self._exit_stack.aclose()

    def reconcile_operations(self) -> str | None:
        """Resolve any install/remove operation a previous boot left at
        ``restart_pending`` — e.g. because the restart it triggered crashed AUG
        before it could record the outcome. Returns a human summary to report
        on the next startup announcement, or None if nothing was pending.
        """
        state = load_state()
        pending = [op for op in state.mcp.operations if op.state == "restart_pending"]
        if not pending:
            return None

        lines = []
        for op in pending:
            health = self.health.get(op.server_name)
            if health is not None and health.status == "active":
                op.state = "active"
                lines.append(
                    f"MCP server '{op.server_name}' {op.action} succeeded "
                    f"({health.tool_count} tools)."
                )
            else:
                op.state = "failed"
                op.detail = health.error if health else "server not found after restart"
                lines.append(f"MCP server '{op.server_name}' {op.action} failed: {op.detail}")
        save_state(state)
        return "\n".join(lines)

    async def _load_one(self, cfg: McpServerConfig) -> None:
        try:
            await asyncio.wait_for(self._connect(cfg), timeout=_PER_SERVER_TIMEOUT)
        except McpSecretError as exc:
            logger.warning("mcp server %s: %s", cfg.name, exc)
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "failed", error=str(exc)
            )
        except TimeoutError:
            logger.warning(
                "mcp server %s: connection timed out after %.0fs", cfg.name, _PER_SERVER_TIMEOUT
            )
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "failed", error="connection timed out"
            )
        except Exception as exc:
            logger.warning("mcp server %s: failed to load: %r", cfg.name, exc)
            self.health[cfg.name] = McpServerHealth(
                cfg.name, cfg.transport, "failed", error=str(exc)
            )

    async def _connect(self, cfg: McpServerConfig) -> None:
        if cfg.transport == "stdio":
            connection = {
                "transport": "stdio",
                "command": cfg.command,
                "args": cfg.args,
                "env": await _resolve_stdio_env(cfg.env),
            }
        else:
            connection = {
                "transport": "streamable_http",
                "url": cfg.url,
                "headers": await _resolve_refs(cfg.headers),
            }

        session = await self._exit_stack.enter_async_context(create_session(connection))
        await session.initialize()
        raw_tools = await load_mcp_tools(session, server_name=cfg.name, tool_name_prefix=False)

        namespaced: list[BaseTool] = []
        for t in raw_tools:
            new_name = f"{cfg.name}__{t.name}"
            if new_name in self._tool_names:
                logger.warning(
                    "mcp server %s: tool name collision on %r, skipped", cfg.name, new_name
                )
                continue
            self._tool_names.add(new_name)
            namespaced.append(_namespace_tool(t, new_name))

        self.tools.extend(namespaced)
        self.health[cfg.name] = McpServerHealth(
            cfg.name, cfg.transport, "active", tool_count=len(namespaced)
        )
        logger.info("mcp server %s: connected, %d tool(s)", cfg.name, len(namespaced))


# ---------------------------------------------------------------------------
# Module-level singleton — tools/mcp.py reads health / triggers restarts
# without needing app.state plumbed through every tool call, same pattern as
# aug/utils/db.py's get_pool()/set_pool().
# ---------------------------------------------------------------------------

_manager: MCPManager | None = None


def set_manager(manager: MCPManager) -> None:
    global _manager
    _manager = manager


def get_manager() -> MCPManager | None:
    """Return the live MCPManager, or None before startup has wired one in."""
    return _manager


# ---------------------------------------------------------------------------
# Durable operation state — install/remove tracked across the restart they
# trigger. See McpOperation / reconcile_operations().
# ---------------------------------------------------------------------------


def record_operation(action: str, server_name: str, state: str) -> str:
    """Persist a new MCP operation record. Returns its id."""
    st = load_state()
    op_id = uuid.uuid4().hex[:8]
    st.mcp.operations.append(
        McpOperation(
            id=op_id, action=action, server_name=server_name, state=state, created_at=time.time()
        )
    )
    save_state(st)
    return op_id


def update_operation_state(op_id: str, state: str, detail: str = "") -> None:
    st = load_state()
    for op in st.mcp.operations:
        if op.id == op_id:
            op.state = state
            op.detail = detail
            break
    save_state(st)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _namespace_tool(t: BaseTool, new_name: str) -> BaseTool:
    """Rename an MCP-derived tool and wrap its call with the per-call timeout."""
    original_coroutine = t.coroutine

    async def _timed(*args, **kwargs):
        return await asyncio.wait_for(
            original_coroutine(*args, **kwargs), timeout=_TOOL_CALL_TIMEOUT
        )

    return StructuredTool(
        name=new_name,
        description=t.description,
        args_schema=t.args_schema,
        coroutine=_timed,
        response_format=t.response_format,
        metadata=t.metadata,
        handle_tool_error=t.handle_tool_error,
    )


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
        raise McpSecretError(f"expected a 'hushed:KEY' reference, got {ref!r}")
    key = ref[len(_HUSHED_PREFIX) :]
    return await asyncio.to_thread(_read_hushed_secret, key)


def _read_hushed_secret(name: str) -> str:
    """Fetch one secret's value from hushed without it ever passing through this
    process's stdout/stderr — ``hushed run`` redacts secret values from the
    wrapped command's output, so printing it to stdout comes back as
    "[REDACTED]" rather than the value. Instead, an extra file descriptor
    pointed at a private temp file is inherited into the wrapped shell, which
    writes the value there directly — a channel hushed never inspects — and we
    read it back once the (bounded, non-interactive) process has exited.

    ``name`` is interpolated into a shell command, so it's restricted to valid
    env-var-name characters first — a config-supplied key name (from settings.json,
    ultimately sourced from an MCP registry listing) must never reach a shell
    unescaped.
    """
    if not _ENV_VAR_NAME_RE.match(name):
        raise McpSecretError(f"invalid secret name {name!r} — must be a valid env var name")

    with tempfile.TemporaryFile() as tf:
        fd = tf.fileno()
        try:
            result = subprocess.run(
                ["hushed", "run", "--", "sh", "-c", f'printf %s "${name}" >&{fd}'],
                pass_fds=(fd,),
                capture_output=True,
                text=True,
                timeout=_HUSHED_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise McpSecretError(f"failed to read secret {name!r}: {exc}") from exc
        if result.returncode != 0:
            raise McpSecretError(f"hushed failed resolving {name!r}: {result.stderr.strip()}")
        tf.seek(0)
        value = tf.read().decode("utf-8", errors="replace")

    if not value:
        raise McpSecretError(f"secret {name!r} is not set. Set it with: hushed add {name} <value>")
    return value
