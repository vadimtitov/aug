"""Restarts AUG (via Portainer) to activate an MCP install/remove, and
delivers the outcome back to whoever asked once it's back up.

install/remove save config and return immediately; the actual restart is
scheduled a few seconds later so the tool's own result has time to reach the
user over Telegram/SSE before the container that would deliver it goes down.
"""

import asyncio
import logging

from aug.config import get_settings
from aug.core.mcp_manager import update_operation_state
from aug.utils.app_ref import get_app
from aug.utils.portainer import PortainerClient

logger = logging.getLogger(__name__)

_RESTART_DELAY_SECONDS = 3

# Keeps delayed-restart tasks alive — asyncio only holds a weak reference to a
# task via create_task(), so without this it could be GC'd mid-sleep.
_background_tasks: set[asyncio.Task] = set()


def schedule_restart(op_id: str, server_name: str, interface: str, thread_id: str) -> None:
    """Fire the restart a few seconds from now, in the background."""

    async def _delayed() -> None:
        await asyncio.sleep(_RESTART_DELAY_SECONDS)
        msg = await trigger_restart(op_id, server_name)
        logger.info("mcp restart op=%s server=%s result=%s", op_id, server_name, msg)
        await _deliver_restart_outcome(interface, thread_id, msg)

    task = asyncio.create_task(_delayed())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def trigger_restart(op_id: str, server_name: str) -> str:
    """Restart AUG via Portainer to activate a saved MCP config change."""
    settings = get_settings()
    container, environment = settings.AUG_CONTAINER, settings.AUG_ENVIRONMENT
    client = PortainerClient()

    if not client.is_configured() or not container or not environment:
        await update_operation_state(op_id, "restart_pending")
        return (
            "Portainer (or AUG_CONTAINER/AUG_ENVIRONMENT) is not configured — restart AUG "
            "manually to activate this change."
        )

    try:
        ep = await client.resolve_endpoint(environment)
        container_id = await client.find_container_id(container, ep["Id"])
        if not container_id:
            detail = f"container '{container}' not found"
            await update_operation_state(op_id, "failed", detail)
            return f"Restart failed: {detail} in '{environment}'."
        # Mark *before* actually pulling the trigger — a successful restart call is
        # meant to kill this very process moments later, before any write made
        # after it could ever land. reconcile_operations() resolves both
        # "restart_pending" and "saved" on the next boot either way.
        await update_operation_state(op_id, "restart_pending")
        await client.container_action(container_id, ep["Id"], "restart")
    except Exception as exc:
        await update_operation_state(op_id, "failed", str(exc))
        return f"Restart failed: {exc}. Restart AUG manually to activate this change."

    return "Restart triggered — AUG will report the result once it's back."


async def _deliver_restart_outcome(interface: str, thread_id: str, message: str) -> None:
    """Best-effort push of a restart outcome back to whoever triggered it."""
    if not (interface and thread_id):
        return
    if interface == "rest_api":
        # AUG is a personal, Telegram-first assistant — REST has no push channel.
        logger.info(
            "mcp restart outcome not delivered: REST has no push channel (thread_id=%s): %s",
            thread_id,
            message,
        )
        return
    app = get_app()
    if app is None:
        return
    iface = getattr(app.state, "interfaces", {}).get(interface)
    if iface is None:
        return
    try:
        actual_thread_id = await iface.resolve_thread(thread_id)
        await iface.send_proactive(actual_thread_id, message)
    except Exception:
        logger.warning(
            "mcp restart outcome delivery failed interface=%s thread_id=%s",
            interface,
            thread_id,
            exc_info=True,
        )
