"""MCP tool support — search, install, list, and remove MCP servers.

Four tools:
  search_mcp_servers(query)  — search the official MCP Registry
  install_mcp_server(index)  — configure + restart to activate (approval required)
  list_mcp_servers()         — show configured servers and their live health
  remove_mcp_server(name)    — remove + restart to deactivate (approval required)

"install #N" resolves against the snapshot from the most recent
search_mcp_servers call, not a fresh search — a search re-run mid-conversation
could otherwise silently reorder what "#2" refers to.

install/remove save config and return immediately; the actual container
restart is scheduled a few seconds later (_schedule_restart) so the tool's own
result has time to reach the user over Telegram/SSE before the container that
would deliver it goes down. The operation is durably recorded first
(record_operation) precisely because that restart can outrace or crash the
process — reconcile_operations() on the next boot resolves it either way.
"""

import asyncio
import logging
import subprocess

from langchain_core.tools import tool

from aug.core.mcp_manager import get_manager, record_operation, update_operation_state
from aug.core.tools.approval import requires_approval
from aug.utils.file_settings import load_settings, save_settings
from aug.utils.mcp_registry import McpRegistryClient, McpRegistryServer
from aug.utils.portainer import PortainerClient

logger = logging.getLogger(__name__)

_AUG_CONTAINER = "aug-aug-1"
_AUG_ENVIRONMENT = "musya"
_RESTART_DELAY_SECONDS = 3
_HUSHED_LIST_TIMEOUT = 10

# "install #N" resolves against this snapshot, not a fresh search. Module-level
# is fine — AUG is a single process serving one user, same as the approval
# rule store and other tool-level state in this codebase.
_last_search: list[McpRegistryServer] = []

# Keeps the delayed-restart task alive — asyncio only holds a weak reference to
# a task via create_task(), so without this the task can be garbage collected
# mid-sleep before it ever fires the restart.
_background_tasks: set[asyncio.Task] = set()


@tool
async def search_mcp_servers(query: str) -> str:
    """Search the official MCP Registry for Model Context Protocol servers.

    Results are numbered and cached — install_mcp_server(index) installs the
    Nth listed result. Re-running a search replaces the cache.

    Args:
        query: Free-text search, e.g. "postgres", "github", "search".
    """
    global _last_search
    client = McpRegistryClient()
    try:
        results = await client.search(query)
    except Exception as exc:
        logger.warning("search_mcp_servers failed query=%r: %r", query, exc)
        return f"MCP Registry search did NOT complete. Error: {exc}"

    _last_search = results
    if not results:
        return f"No MCP servers found for {query!r}."

    lines = [f"Found {len(results)} result(s):"]
    for i, s in enumerate(results, 1):
        transport_desc = "http" if s.transport == "http" else f"stdio ({s.command})"
        lines.append(f"  {i}. {s.name} ({s.namespace})\n     {s.description} | {transport_desc}")
    return "\n".join(lines)


@tool
def list_mcp_servers() -> str:
    """List configured MCP servers and their live connection health."""
    configured = load_settings().mcp_servers
    if not configured:
        return "No MCP servers configured. Use search_mcp_servers to find one."

    manager = get_manager()
    lines = ["Configured MCP servers:"]
    for cfg in configured:
        if not cfg.enabled:
            status = "disabled"
        else:
            health = manager.health.get(cfg.name) if manager else None
            if health is None:
                status = "pending restart"
            elif health.status == "active":
                status = f"active, {health.tool_count} tools"
            else:
                status = f"failed ({health.error})"
        lines.append(f"  {cfg.name} — {cfg.transport}, {status}")
    return "\n".join(lines)


@tool
@requires_approval(describe=lambda index: _describe_install(index))
async def install_mcp_server(index: int) -> str:
    """Install an MCP server found by a previous search_mcp_servers call.

    If the server needs secrets (API keys, tokens, connection strings) that
    aren't set yet, this returns an install plan instead of installing —
    tell the user what to set on musya (`hushed add NAME <value>`), then call
    this again with the same index once they've done so.

    On success, saves the server config and restarts AUG to activate it.

    Args:
        index: The 1-based result number from the last search_mcp_servers call.
    """
    if not _last_search:
        return "Run search_mcp_servers first, then install by result number."
    if not (1 <= index <= len(_last_search)):
        return f"No result #{index}. The last search had {len(_last_search)} result(s)."
    server = _last_search[index - 1]

    settings = load_settings()
    if any(s.name == server.slug for s in settings.mcp_servers):
        return f"'{server.slug}' is already configured. Use remove_mcp_server to reinstall."

    known_secrets = await asyncio.to_thread(_list_hushed_secrets)
    missing = [name for name in server.required_env if name not in known_secrets]
    if missing:
        return _install_plan(server, index, missing)

    env_refs = {name: f"hushed:{name}" for name in server.required_env}
    cfg = server.to_config(env_refs)
    settings.mcp_servers.append(cfg)
    save_settings(settings)

    op_id = record_operation("install", cfg.name, "saved")
    _schedule_restart(op_id, cfg.name)
    return f"Config saved for '{cfg.name}'. Restarting to activate..."


@tool
@requires_approval(describe=lambda name: (name, "remove MCP server"))
async def remove_mcp_server(name: str) -> str:
    """Remove a configured MCP server and restart to deactivate it.

    Args:
        name: The configured server's name (see list_mcp_servers).
    """
    settings = load_settings()
    remaining = [s for s in settings.mcp_servers if s.name != name]
    if len(remaining) == len(settings.mcp_servers):
        return f"No MCP server named '{name}' is configured. See list_mcp_servers for what's there."
    settings.mcp_servers = remaining
    save_settings(settings)

    op_id = record_operation("remove", name, "saved")
    _schedule_restart(op_id, name)
    return f"Removed '{name}' from config. Restarting to deactivate..."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _describe_install(index: int) -> tuple[str, str]:
    if _last_search and 1 <= index <= len(_last_search):
        s = _last_search[index - 1]
        return (s.name, f"install MCP server {s.name} ({s.transport})")
    return (f"#{index}", "install MCP server")


def _install_plan(server: McpRegistryServer, index: int, missing: list[str]) -> str:
    lines = [f"Install plan: {server.name} ({server.transport})"]
    if server.transport == "stdio":
        lines.append(f"Command: {server.command} {' '.join(server.args)}")
    else:
        lines.append(f"URL: {server.url}")
    lines.append("Required secrets not yet set:")
    for name in missing:
        lines.append(f"  hushed add {name} <value>")
    lines.append(f"Set them on musya, then install result #{index} again.")
    return "\n".join(lines)


def _list_hushed_secrets() -> set[str]:
    """Names only — hushed never reveals values via `list`, so this check never
    exposes anything to AUG that it shouldn't see."""
    try:
        result = subprocess.run(
            ["hushed", "list"], capture_output=True, text=True, timeout=_HUSHED_LIST_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("hushed list failed: %r", exc)
        return set()
    if result.returncode != 0:
        logger.warning("hushed list exit_code=%d stderr=%.200r", result.returncode, result.stderr)
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


async def _trigger_restart(op_id: str, server_name: str) -> str:
    client = PortainerClient()
    if not client.is_configured():
        update_operation_state(op_id, "restart_pending")
        return (
            "Portainer is not configured — restart AUG manually to activate this "
            f"change (e.g. `docker restart {_AUG_CONTAINER}` on {_AUG_ENVIRONMENT})."
        )
    try:
        ep = await client.resolve_endpoint(_AUG_ENVIRONMENT)
        container_id = await client.find_container_id(_AUG_CONTAINER, ep["Id"])
        if not container_id:
            detail = f"container '{_AUG_CONTAINER}' not found"
            update_operation_state(op_id, "failed", detail)
            return f"Restart failed: {detail} in '{_AUG_ENVIRONMENT}'."
        await client.container_action(container_id, ep["Id"], "restart")
    except Exception as exc:
        update_operation_state(op_id, "failed", str(exc))
        return f"Restart failed: {exc}. Restart AUG manually to activate this change."

    update_operation_state(op_id, "restart_pending")
    return "Restart triggered — AUG will report the result once it's back."


def _schedule_restart(op_id: str, server_name: str) -> None:
    """Fire the restart a few seconds after returning, so the tool's own result
    reaches the user before the container that would deliver it goes down."""

    async def _delayed() -> None:
        await asyncio.sleep(_RESTART_DELAY_SECONDS)
        msg = await _trigger_restart(op_id, server_name)
        logger.info("mcp restart op=%s server=%s result=%s", op_id, server_name, msg)

    _background_tasks.add(task := asyncio.create_task(_delayed()))
    task.add_done_callback(_background_tasks.discard)
