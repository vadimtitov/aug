"""Portainer management tools via Portainer REST API."""

import logging
from typing import Literal

import httpx
from langchain_core.tools import tool

from aug.utils.portainer import PortainerClient, strip_docker_log_headers

logger = logging.getLogger(__name__)

_NOT_CONFIGURED = "Portainer is not configured: PORTAINER_URL and PORTAINER_API_TOKEN are required."


@tool
async def portainer_list_containers() -> str:
    """List all Docker containers with their current status.

    Returns container names, image, status, and uptime.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED
    try:
        containers = await client.list_containers()
    except httpx.HTTPStatusError as e:
        return f"Portainer error: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not containers:
        return "No containers found."

    lines = []
    for c in containers:
        name = ", ".join(n.lstrip("/") for n in c.get("Names", ["?"]))
        lines.append(
            f"• {name}  [{c.get('State', '?')}] {c.get('Status', '?')}"
            f"  image={c.get('Image', '?')}  id={c.get('Id', '')[:12]}"
        )
    logger.info("portainer_list_containers: %d containers", len(lines))
    return f"{len(lines)} containers:\n" + "\n".join(lines)


@tool
async def portainer_container_logs(container: str, tail: int = 100) -> str:
    """Get recent logs from a Docker container.

    Args:
        container: Container name or ID (partial IDs accepted).
        tail: Number of log lines to return (default 100, max 500).
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    tail = min(tail, 500)
    try:
        container_id = await client.find_container_id(container)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not container_id:
        return f"Container not found: {container!r}"

    try:
        raw = await client.container_logs(container_id, tail)
    except httpx.HTTPStatusError as e:
        return f"Portainer error: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    text = strip_docker_log_headers(raw)
    logger.info("portainer_container_logs container=%r lines=%d", container, text.count("\n"))
    return text or "(no log output)"


@tool
async def portainer_container_action(
    container: str,
    action: Literal["start", "stop", "restart", "remove"],
) -> str:
    """Perform a lifecycle action on a Docker container.

    Args:
        container: Container name or ID (partial IDs accepted).
        action:    One of: start, stop, restart, remove.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        container_id = await client.find_container_id(container)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not container_id:
        return f"Container not found: {container!r}"

    try:
        await client.container_action(container_id, action)
    except httpx.HTTPStatusError as e:
        return f"Action '{action}' failed: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    logger.info("portainer_container_action container=%r action=%s", container, action)
    return f"Container {container!r}: {action} succeeded."


@tool
async def portainer_list_stacks() -> str:
    """List all stacks (docker-compose deployments) in Portainer with their status."""
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        stacks = await client.list_stacks()
    except httpx.HTTPStatusError as e:
        return f"Portainer error: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not stacks:
        return "No stacks found."

    lines = []
    for s in stacks:
        status_label = "active" if s.get("Status") == 1 else "inactive"
        lines.append(f"• {s.get('Name', '?')}  [{status_label}]  id={s.get('Id', '?')}")
    logger.info("portainer_list_stacks: %d stacks", len(lines))
    return f"{len(lines)} stacks:\n" + "\n".join(lines)


@tool
async def portainer_deploy_stack(name: str, compose: str) -> str:
    """Deploy a new stack (docker-compose) to Portainer, or update an existing one.

    Use this to deploy services — media servers, databases, monitoring tools, etc.
    If a stack with the given name already exists it will be updated with the new compose.

    Args:
        name:    Stack name (lowercase, no spaces, e.g. "jellyfin", "my-media-server").
        compose: Full docker-compose.yml content as a string.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        data, action = await client.deploy_stack(name, compose)
    except httpx.HTTPStatusError as e:
        return f"Stack operation failed: HTTP {e.response.status_code} — {e.response.text[:300]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    stack_id = data.get("Id", "?")
    logger.info("portainer_deploy_stack name=%r action=%s id=%s", name, action, stack_id)
    return f"Stack {name!r} {action} successfully (id={stack_id})."


@tool
async def portainer_stack_action(
    name: str,
    action: Literal["start", "stop"],
) -> str:
    """Perform a lifecycle action on a Portainer stack.

    Args:
        name:   Stack name.
        action: One of: start, stop.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        stack = await client.find_stack(name)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not stack:
        return f"Stack not found: {name!r}"

    stack_id = stack["Id"]
    try:
        await client.stack_action(stack_id, action)
    except httpx.HTTPStatusError as e:
        return f"Stack '{action}' failed: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    logger.info("portainer_stack_action name=%r action=%s id=%s", name, action, stack_id)
    return f"Stack {name!r}: {action} succeeded."


# ---------------------------------------------------------------------------
# Kept for backward compatibility with agents that reference this directly.
# New agents should use portainer_container_action instead.
# ---------------------------------------------------------------------------


@tool
async def portainer_restart_container(container: str) -> str:
    """Restart a Docker container.

    Args:
        container: Container name or ID (partial IDs accepted).
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        container_id = await client.find_container_id(container)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not container_id:
        return f"Container not found: {container!r}"

    try:
        await client.container_action(container_id, "restart")
    except httpx.HTTPStatusError as e:
        return f"Restart failed: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    logger.info("portainer_restart_container container=%r id=%s", container, container_id[:12])
    return f"Container {container!r} restarted successfully."
