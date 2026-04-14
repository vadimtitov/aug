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
    """List all Docker containers across all Portainer environments, grouped by environment.

    Use this to discover available environments and their containers before
    performing any targeted operations.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED
    try:
        endpoints = await client.list_endpoints()
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not endpoints:
        return "No environments found in Portainer."

    sections = []
    total = 0
    for ep in endpoints:
        ep_id = ep["Id"]
        ep_name = ep.get("Name", str(ep_id))
        try:
            containers = await client.list_containers(ep_id)
        except httpx.HTTPStatusError as e:
            sections.append(f"**{ep_name}**\n  Error: HTTP {e.response.status_code}")
            continue
        except httpx.RequestError as e:
            sections.append(f"**{ep_name}**\n  Unreachable: {e}")
            continue

        if not containers:
            sections.append(f"**{ep_name}**\n  (no containers)")
            continue

        lines = []
        for c in containers:
            name = ", ".join(n.lstrip("/") for n in c.get("Names", ["?"]))
            lines.append(
                f"  • {name}  [{c.get('State', '?')}] {c.get('Status', '?')}"
                f"  image={c.get('Image', '?')}  id={c.get('Id', '')[:12]}"
            )
        total += len(lines)
        sections.append(f"**{ep_name}**\n" + "\n".join(lines))

    logger.info(
        "portainer_list_containers: %d containers across %d environments", total, len(endpoints)
    )
    return "\n\n".join(sections)


@tool
async def portainer_container_logs(container: str, environment: str, tail: int = 100) -> str:
    """Get recent logs from a Docker container.

    Args:
        container:   Container name or ID (partial IDs accepted).
        environment: Portainer environment name (e.g. "musya", "dusya").
                     Use portainer_list_containers to discover available environments.
        tail:        Number of log lines to return (default 100, max 500).
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    tail = min(tail, 500)
    try:
        ep = await client.resolve_endpoint(environment)
    except ValueError as e:
        return str(e)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    ep_id = ep["Id"]
    try:
        container_id = await client.find_container_id(container, ep_id)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not container_id:
        return f"Container {container!r} not found in environment {environment!r}."

    try:
        raw = await client.container_logs(container_id, ep_id, tail)
    except httpx.HTTPStatusError as e:
        return f"Portainer error: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    text = strip_docker_log_headers(raw)
    logger.info(
        "portainer_container_logs container=%r env=%r lines=%d",
        container,
        environment,
        text.count("\n"),
    )
    return text or "(no log output)"


@tool
async def portainer_container_action(
    container: str,
    action: Literal["start", "stop", "restart", "remove"],
    environment: str,
) -> str:
    """Perform a lifecycle action on a Docker container.

    Args:
        container:   Container name or ID (partial IDs accepted).
        action:      One of: start, stop, restart, remove.
        environment: Portainer environment name (e.g. "musya", "dusya").
                     Use portainer_list_containers to discover available environments.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        ep = await client.resolve_endpoint(environment)
    except ValueError as e:
        return str(e)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    ep_id = ep["Id"]
    try:
        container_id = await client.find_container_id(container, ep_id)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not container_id:
        return f"Container {container!r} not found in environment {environment!r}."

    try:
        await client.container_action(container_id, ep_id, action)
    except httpx.HTTPStatusError as e:
        return f"Action '{action}' failed: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    logger.info(
        "portainer_container_action container=%r env=%r action=%s", container, environment, action
    )
    return f"Container {container!r} in {environment!r}: {action} succeeded."


@tool
async def portainer_list_stacks() -> str:
    """List all stacks (docker-compose deployments) across all Portainer environments.

    Use this to discover available environments and their stacks before
    performing any targeted operations.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        endpoints = await client.list_endpoints()
        stacks = await client.list_stacks()
    except httpx.HTTPStatusError as e:
        return f"Portainer error: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not stacks:
        return "No stacks found."

    ep_names = {ep["Id"]: ep.get("Name", str(ep["Id"])) for ep in endpoints}
    by_env: dict[str, list[str]] = {}
    for s in stacks:
        ep_name = ep_names.get(s.get("EndpointId", -1), "unknown")
        status_label = "active" if s.get("Status") == 1 else "inactive"
        line = f"  • {s.get('Name', '?')}  [{status_label}]  id={s.get('Id', '?')}"
        by_env.setdefault(ep_name, []).append(line)

    sections = [f"**{env}**\n" + "\n".join(lines) for env, lines in by_env.items()]
    logger.info("portainer_list_stacks: %d stacks across %d environments", len(stacks), len(by_env))
    return "\n\n".join(sections)


@tool
async def portainer_deploy_stack(name: str, compose: str, environment: str) -> str:
    """Deploy a new stack (docker-compose) to a Portainer environment, or update an existing one.

    Use this to deploy services — media servers, databases, monitoring tools, etc.
    If a stack with the given name already exists in the environment it will be updated.

    Args:
        name:        Stack name (lowercase, no spaces, e.g. "jellyfin", "my-media-server").
        compose:     Full docker-compose.yml content as a string.
        environment: Portainer environment name to deploy to (e.g. "musya", "dusya").
                     Use portainer_list_stacks to discover available environments.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        ep = await client.resolve_endpoint(environment)
    except ValueError as e:
        return str(e)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    try:
        data, action = await client.deploy_stack(name, compose, ep["Id"])
    except httpx.HTTPStatusError as e:
        return f"Stack operation failed: HTTP {e.response.status_code} — {e.response.text[:300]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    stack_id = data.get("Id", "?")
    logger.info(
        "portainer_deploy_stack name=%r env=%r action=%s id=%s", name, environment, action, stack_id
    )
    return f"Stack {name!r} {action} successfully in {environment!r} (id={stack_id})."


@tool
async def portainer_stack_action(
    name: str,
    action: Literal["start", "stop"],
    environment: str,
) -> str:
    """Perform a lifecycle action on a Portainer stack.

    Args:
        name:        Stack name.
        action:      One of: start, stop.
        environment: Portainer environment name (e.g. "musya", "dusya").
                     Use portainer_list_stacks to discover available environments.
    """
    client = PortainerClient()
    if not client.is_configured():
        return _NOT_CONFIGURED

    try:
        ep = await client.resolve_endpoint(environment)
    except ValueError as e:
        return str(e)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    ep_id = ep["Id"]
    try:
        stack = await client.find_stack(name, ep_id)
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    if not stack:
        return f"Stack {name!r} not found in environment {environment!r}."

    stack_id = stack["Id"]
    try:
        await client.stack_action(stack_id, ep_id, action)
    except httpx.HTTPStatusError as e:
        return f"Stack '{action}' failed: HTTP {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Portainer unreachable: {e}"

    logger.info(
        "portainer_stack_action name=%r env=%r action=%s id=%s", name, environment, action, stack_id
    )
    return f"Stack {name!r} in {environment!r}: {action} succeeded."
