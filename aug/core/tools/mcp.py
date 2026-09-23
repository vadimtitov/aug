"""MCP tool support — search, install, list, and remove MCP servers.

Four tools:
  search_mcp_servers(query)  — search the official MCP Registry
  install_mcp_server(index)  — configure + restart to activate (approval required)
  list_mcp_servers()         — show configured servers and their live health
  remove_mcp_server(name)    — remove + restart to deactivate (approval required)

"install #N" resolves against the search snapshot for the *current
conversation* (see ``_current_thread_id``), not a global one — a search
someone runs in another conversation can never reorder what "#2" refers to
here.

The moment an index first resolves to a real result, it's frozen into an
immutable ``McpInstallPlan`` (server identity, pinned version, and exactly
which secret backs each credential) and persisted to state.json, keyed by
thread — before the approval interrupt ever pauses the graph. Approval binds
to that plan, not to a live, mutable index: replaying the tool call on resume
(LangGraph re-executes the node from the top) looks the plan up again rather
than re-deriving it, so it survives both a concurrent search elsewhere and a
process restart while approval is pending.

install/remove save config and return immediately; the actual container
restart is scheduled a few seconds later (_schedule_restart) so the tool's own
result has time to reach the user over Telegram/SSE before the container that
would deliver it goes down. The operation is durably recorded first
(record_operation) precisely because that restart can outrace or crash the
process — reconcile_operations() on the next boot resolves it either way, and
delivers the outcome back to the conversation that requested it.
"""

import asyncio
import logging
import re
import subprocess
import time
import uuid

from langchain_core.tools import tool
from langgraph.config import get_config

from aug.core.mcp_manager import get_manager, record_operation, update_operation_state
from aug.core.prompts import (
    MCP_INSTALL_PLAN_CREDENTIAL_BOUND,
    MCP_INSTALL_PLAN_CREDENTIAL_MISSING,
    MCP_INSTALL_PLAN_CREDENTIALS_HEADER,
    MCP_INSTALL_PLAN_DESTINATION_HTTP,
    MCP_INSTALL_PLAN_DESTINATION_STDIO,
    MCP_INSTALL_PLAN_HEADER,
    MCP_INSTALL_PLAN_MISSING_FOOTER,
    MCP_INSTALL_PLAN_NO_CREDENTIALS,
)
from aug.core.tools.approval import requires_approval
from aug.utils.file_settings import McpServerConfig, load_settings, update_settings
from aug.utils.mcp_registry import McpRegistryClient, McpRegistryServer
from aug.utils.portainer import PortainerClient
from aug.utils.state import McpCredentialBinding, McpInstallPlan, load_state, save_state

logger = logging.getLogger(__name__)

_AUG_CONTAINER = "aug-aug-1"
_AUG_ENVIRONMENT = "musya"
_RESTART_DELAY_SECONDS = 3
_HUSHED_LIST_TIMEOUT = 10
# Turns a target input name (an env var or an HTTP header, e.g. "X-API-Key")
# into a valid hushed secret identifier — the two are never the same thing,
# since header names routinely contain characters hushed secret names can't.
_SECRET_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")


class _McpToolState:
    """Object-owned state for these tools — an explicit instance instead of
    bare module globals, per the codebase's state-belongs-to-objects rule.

    ``last_search`` is scoped per conversation (thread_id) so one
    conversation's search can never be resolved against another's "#N".
    ``background_tasks`` keeps delayed-restart tasks alive — asyncio only
    holds a weak reference to a task via create_task(), so without this the
    task could be garbage collected mid-sleep before it ever fires.
    """

    def __init__(self) -> None:
        self.last_search: dict[str, list[McpRegistryServer]] = {}
        self.background_tasks: set[asyncio.Task] = set()


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
        return f"MCP Registry search did NOT complete. Error: {exc}"

    _state.last_search[_current_thread_id()] = results
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
    thread_id = _current_thread_id()
    plan = _get_or_build_plan(thread_id, index)
    if plan is None:
        snapshot = _state.last_search.get(thread_id)
        if not snapshot:
            return "Run search_mcp_servers first, then install by result number."
        return f"No result #{index}. The last search had {len(snapshot)} result(s)."

    if any(s.name == plan.slug for s in load_settings().mcp_servers):
        _clear_plan(thread_id)
        return f"'{plan.slug}' is already configured. Use remove_mcp_server to reinstall."

    missing = [c.target_name for c in plan.credentials if not c.bound]
    if missing:
        return _render_missing_secrets(plan)

    bindings = {c.target_name: f"hushed:{c.secret_name}" for c in plan.credentials}
    async with update_settings() as settings:
        if any(s.name == plan.slug for s in settings.mcp_servers):
            _clear_plan(thread_id)
            return f"'{plan.slug}' is already configured. Use remove_mcp_server to reinstall."
        settings.mcp_servers.append(_plan_to_config(plan, bindings))

    _clear_plan(thread_id)
    op_id = await record_operation(
        "install", plan.slug, "saved", interface=_current_interface(), thread_id=thread_id
    )
    _schedule_restart(op_id, plan.slug)
    return f"Config saved for '{plan.slug}'. Restarting to activate..."


@tool
@requires_approval(describe=lambda name: (name, "remove MCP server"))
async def remove_mcp_server(name: str) -> str:
    """Remove a configured MCP server and restart to deactivate it.

    Args:
        name: The configured server's name (see list_mcp_servers).
    """
    found = False
    async with update_settings() as settings:
        remaining = [s for s in settings.mcp_servers if s.name != name]
        found = len(remaining) != len(settings.mcp_servers)
        if found:
            settings.mcp_servers = remaining

    if not found:
        return f"No MCP server named '{name}' is configured. See list_mcp_servers for what's there."

    op_id = await record_operation(
        "remove", name, "saved", interface=_current_interface(), thread_id=_current_thread_id()
    )
    _schedule_restart(op_id, name)
    return f"Removed '{name}' from config. Restarting to deactivate..."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _current_thread_id() -> str:
    """The LangGraph thread this tool call is running in, or "" outside a
    runnable context (e.g. a tool invoked directly in a unit test)."""
    return _configurable("thread_id")


def _current_interface() -> str:
    return _configurable("interface")


def _configurable(key: str) -> str:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ""
    return configurable.get(key) or ""


def _describe_install(index: int) -> tuple[str, str]:
    plan = _get_or_build_plan(_current_thread_id(), index)
    if plan is None:
        return (f"#{index}", "install MCP server")
    return (plan.slug, _render_plan(plan))


def _get_or_build_plan(thread_id: str, index: int) -> McpInstallPlan | None:
    """Return the durable plan for *index* in this conversation, building and
    persisting one if this is the first time it's been resolved.

    Reusing an existing plan (rather than re-deriving it from the live search
    snapshot) is what makes approval replay-safe: the decorator calls this
    again on every resume, and a plan already on disk answers identically
    regardless of what ``search_mcp_servers`` has done elsewhere since, or
    whether this process even restarted in between.
    """
    state = load_state()
    existing = state.mcp.install_plans.get(thread_id)
    if existing is not None and existing.search_index == index:
        return existing

    snapshot = _state.last_search.get(thread_id) or []
    if not (1 <= index <= len(snapshot)):
        return None

    server = snapshot[index - 1]
    known_secrets = _list_hushed_secrets()
    plan = _build_plan(thread_id, index, server, known_secrets)
    state.mcp.install_plans[thread_id] = plan
    save_state(state)
    return plan


def _build_plan(
    thread_id: str, index: int, server: McpRegistryServer, known_secrets: set[str]
) -> McpInstallPlan:
    credentials = []
    for name in server.required_inputs:
        secret_name = _secret_name_for(name)
        credentials.append(
            McpCredentialBinding(
                target_name=name, secret_name=secret_name, bound=secret_name in known_secrets
            )
        )
    return McpInstallPlan(
        id=uuid.uuid4().hex[:8],
        thread_id=thread_id,
        search_index=index,
        server_name=server.name,
        slug=server.slug,
        version=server.version,
        transport=server.transport,
        command=server.command,
        args=server.args,
        url=server.url,
        credentials=credentials,
        created_at=time.time(),
    )


def _secret_name_for(target_name: str) -> str:
    """Sanitize a requested input name into a valid hushed secret identifier —
    e.g. header "X-API-Key" -> "X_API_KEY". Env var names already qualify and
    pass through unchanged."""
    name = _SECRET_NAME_SANITIZE_RE.sub("_", target_name).upper()
    return f"_{name}" if not name or name[0].isdigit() else name


def _plan_to_config(plan: McpInstallPlan, bindings: dict[str, str]) -> McpServerConfig:
    if plan.transport == "stdio":
        return McpServerConfig(
            name=plan.slug,
            transport="stdio",
            command=plan.command,
            args=plan.args,
            env=bindings,
            enabled=True,
        )
    return McpServerConfig(
        name=plan.slug, transport="http", url=plan.url, headers=bindings, enabled=True
    )


def _clear_plan(thread_id: str) -> None:
    state = load_state()
    if state.mcp.install_plans.pop(thread_id, None) is not None:
        save_state(state)


def _render_plan(plan: McpInstallPlan) -> str:
    lines = [
        MCP_INSTALL_PLAN_HEADER.format(
            name=plan.server_name, version=plan.version, transport=plan.transport
        )
    ]
    if plan.transport == "stdio":
        lines.append(
            MCP_INSTALL_PLAN_DESTINATION_STDIO.format(
                command=plan.command, args=" ".join(plan.args)
            )
        )
    else:
        lines.append(MCP_INSTALL_PLAN_DESTINATION_HTTP.format(url=plan.url))

    if not plan.credentials:
        lines.append(MCP_INSTALL_PLAN_NO_CREDENTIALS)
    else:
        lines.append(MCP_INSTALL_PLAN_CREDENTIALS_HEADER)
        for c in plan.credentials:
            template = (
                MCP_INSTALL_PLAN_CREDENTIAL_BOUND
                if c.bound
                else MCP_INSTALL_PLAN_CREDENTIAL_MISSING
            )
            lines.append(template.format(target_name=c.target_name, secret_name=c.secret_name))
    return "\n".join(lines)


def _render_missing_secrets(plan: McpInstallPlan) -> str:
    return "\n".join(
        [_render_plan(plan), MCP_INSTALL_PLAN_MISSING_FOOTER.format(index=plan.search_index)]
    )


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
        await update_operation_state(op_id, "restart_pending")
        return (
            "Portainer is not configured — restart AUG manually to activate this "
            f"change (e.g. `docker restart {_AUG_CONTAINER}` on {_AUG_ENVIRONMENT})."
        )
    try:
        ep = await client.resolve_endpoint(_AUG_ENVIRONMENT)
        container_id = await client.find_container_id(_AUG_CONTAINER, ep["Id"])
        if not container_id:
            detail = f"container '{_AUG_CONTAINER}' not found"
            await update_operation_state(op_id, "failed", detail)
            return f"Restart failed: {detail} in '{_AUG_ENVIRONMENT}'."
        # Mark *before* actually pulling the trigger — a successful restart
        # call is meant to kill this very process moments later, before any
        # write made after it could ever land. reconcile_operations() resolves
        # both "restart_pending" and "saved" on the next boot, so either
        # ordering is recoverable, but marking first keeps the durable state
        # honest about what was actually attempted.
        await update_operation_state(op_id, "restart_pending")
        await client.container_action(container_id, ep["Id"], "restart")
    except Exception as exc:
        await update_operation_state(op_id, "failed", str(exc))
        return f"Restart failed: {exc}. Restart AUG manually to activate this change."

    return "Restart triggered — AUG will report the result once it's back."


def _schedule_restart(op_id: str, server_name: str) -> None:
    """Fire the restart a few seconds after returning, so the tool's own result
    reaches the user before the container that would deliver it goes down."""

    async def _delayed() -> None:
        await asyncio.sleep(_RESTART_DELAY_SECONDS)
        msg = await _trigger_restart(op_id, server_name)
        logger.info("mcp restart op=%s server=%s result=%s", op_id, server_name, msg)

    task = asyncio.create_task(_delayed())
    _state.background_tasks.add(task)
    task.add_done_callback(_state.background_tasks.discard)
