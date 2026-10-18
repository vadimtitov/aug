"""MCP tool support — search, install, list, and remove MCP servers.

Install planning lives in aug.core.mcp.install, restart delivery in
aug.core.mcp.restart, hushed secret listing in aug.utils.hushed — this module
is the tool surface plus the per-conversation search-result cache that ties
them together ("install #N" resolves against *this* conversation's last
search, never another conversation's).
"""

import logging

from langchain_core.tools import tool

from aug.core.mcp import install as mcp_install
from aug.core.mcp.manager import get_manager, record_operation, update_operation_state
from aug.core.mcp.restart import can_report_outcome, can_self_restart, schedule_restart
from aug.core.tools.approval import requires_approval
from aug.core.tools.context import current_interface, current_thread_id
from aug.utils.file_settings import load_settings, update_settings
from aug.utils.mcp_registry import McpRegistryClient, McpRegistryServer

logger = logging.getLogger(__name__)


class _McpToolState:
    """Per-conversation search-result cache — an explicit instance instead of
    a bare module global, per the codebase's state-belongs-to-objects rule."""

    def __init__(self) -> None:
        self.last_search: dict[str, list[McpRegistryServer]] = {}


_state = _McpToolState()


@tool
async def search_mcp_servers(query: str) -> str:
    """Search the official MCP Registry for Model Context Protocol servers.

    Results are numbered and cached for this conversation — install_mcp_server(index)
    installs the Nth listed result. Re-running a search replaces the cache.

    Args:
        query: Free-text search, e.g. "postgres", "github", "search".
    """
    client = McpRegistryClient()
    try:
        results = await client.search(query)
    except Exception as exc:
        logger.warning("search_mcp_servers failed query=%r: %r", query, exc)
        # type name too: some httpx errors (e.g. ReadTimeout) stringify to "".
        return f"MCP Registry search did NOT complete. Error: {type(exc).__name__}: {exc}"

    _state.last_search[current_thread_id()] = results
    if not results:
        return f"No MCP servers found for {query!r}."

    lines = [f"Found {len(results)} result(s):"]
    for i, s in enumerate(results, 1):
        transport_desc = "http" if s.transport == "http" else f"stdio ({s.command})"
        lines.append(f"  {i}. {s.name} ({s.namespace})\n     {s.description} | {transport_desc}")
    return "\n".join(lines)


@tool
async def list_mcp_servers() -> str:
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
@requires_approval(
    describe=lambda index: _describe_install(index),
    on_denied=lambda index: mcp_install.clear_plan(current_thread_id()),
)
async def install_mcp_server(index: int) -> str:
    """Install an MCP server found by a previous search_mcp_servers call.

    If the server needs secrets that aren't set yet, this returns an install plan
    instead of installing — set them with `hushed add NAME <value>`, then call this
    again with the same index. On success, saves the config and restarts AUG.

    Args:
        index: The 1-based result number from the last search_mcp_servers call.
    """
    thread_id = current_thread_id()
    snapshot = _state.last_search.get(thread_id, [])
    plan = await mcp_install.get_or_build_plan(thread_id, index, snapshot)
    if plan is None:
        if not snapshot:
            return "Run search_mcp_servers first, then install by result number."
        return f"No result #{index}. The last search had {len(snapshot)} result(s)."

    if any(s.name == plan.slug for s in load_settings().mcp_servers):
        mcp_install.clear_plan(thread_id)
        return f"'{plan.slug}' is already configured. Use remove_mcp_server to reinstall."

    missing = [c.target_name for c in plan.credentials if not c.bound]
    if missing:
        return mcp_install.render_missing_secrets(plan)

    interface = current_interface()
    bindings = {c.target_name: f"hushed:{c.secret_name}" for c in plan.credentials}

    # Operation record first, config second: if AUG dies between the two writes, an
    # operation with no matching config safely reads as "failed" on the next
    # reconcile; the reverse order leaves a config change nothing can reconcile.
    op_id = await record_operation(
        "install", plan.slug, "saved", interface=interface, thread_id=thread_id
    )

    async with update_settings() as settings:
        if any(s.name == plan.slug for s in settings.mcp_servers):
            mcp_install.clear_plan(thread_id)
            await update_operation_state(op_id, "failed", "server already configured")
            return f"'{plan.slug}' is already configured. Use remove_mcp_server to reinstall."
        settings.mcp_servers.append(mcp_install.plan_to_config(plan, bindings))

    mcp_install.clear_plan(thread_id)
    schedule_restart(op_id, plan.slug, interface, thread_id)
    return _pending_message(
        f"Config saved for '{plan.slug}', but it is NOT active yet — its tools only "
        "appear after AUG restarts.",
        "install",
        interface,
    )


@tool
@requires_approval(describe=lambda name: (name, "remove MCP server"))
async def remove_mcp_server(name: str) -> str:
    """Remove a configured MCP server and restart to deactivate it.

    Args:
        name: The configured server's name (see list_mcp_servers).
    """
    if not any(s.name == name for s in load_settings().mcp_servers):
        return f"No MCP server named '{name}' is configured. See list_mcp_servers for what's there."

    interface = current_interface()
    thread_id = current_thread_id()
    # Operation record first, config second — see install_mcp_server's comment.
    op_id = await record_operation(
        "remove", name, "saved", interface=interface, thread_id=thread_id
    )

    found = False
    async with update_settings() as settings:
        remaining = [s for s in settings.mcp_servers if s.name != name]
        found = len(remaining) != len(settings.mcp_servers)
        if found:
            settings.mcp_servers = remaining

    if not found:
        # Lost a race against another removal between the check above and the lock.
        await update_operation_state(op_id, "failed", "server no longer configured")
        return f"No MCP server named '{name}' is configured. See list_mcp_servers for what's there."

    schedule_restart(op_id, name, interface, thread_id)
    return _pending_message(
        f"Removed '{name}' from config, but its tools stay active until AUG restarts.",
        "removal",
        interface,
    )


async def _describe_install(index: int) -> tuple[str, str]:
    """Approval-preview glue: bind the decorator's index to this conversation's plan."""
    thread_id = current_thread_id()
    snapshot = _state.last_search.get(thread_id, [])
    return await mcp_install.describe_plan(thread_id, index, snapshot)


def _pending_message(saved: str, what: str, interface: str) -> str:
    """Install/remove result: what was saved, plus only the follow-up that will
    actually happen — a restart and an outcome report are both conditional."""
    restart = (
        "AUG will attempt to restart itself in a few seconds."
        if can_self_restart()
        else "AUG will NOT restart by itself (Portainer / AUG_CONTAINER / AUG_ENVIRONMENT "
        "not configured) — the user must restart it manually."
    )
    report = (
        "The outcome is reported here once AUG is back."
        if can_report_outcome(interface)
        else "No outcome will be reported here — check list_mcp_servers after the restart."
    )
    return f"{saved} {restart} {report} Tell the user the {what} is pending, not done."
