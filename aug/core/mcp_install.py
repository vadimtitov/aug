"""MCP install-plan resolution — turns a search result into an immutable,
persisted plan an approval decision can bind to, and back into a saved config.

The moment an index first resolves to a real result, it's frozen (server identity,
pinned version, exactly which secret backs each credential) and persisted to
state.json, keyed by thread — before the approval interrupt ever pauses the graph.
Replaying the tool call on resume looks the plan up again rather than re-deriving
it, so it survives both a concurrent search elsewhere and a process restart while
approval is pending. Every lookup still revalidates each credential's ``bound``
status against hushed's current secret list, so a plan built while a secret was
missing doesn't say so forever once the user adds it.
"""

import re
import time
import uuid

from aug.core.prompts import (
    MCP_INSTALL_PLAN_CREDENTIAL_BOUND,
    MCP_INSTALL_PLAN_CREDENTIAL_MISSING,
    MCP_INSTALL_PLAN_CREDENTIALS_HEADER,
    MCP_INSTALL_PLAN_DEFAULT_LINE,
    MCP_INSTALL_PLAN_DEFAULTS_HEADER,
    MCP_INSTALL_PLAN_DESTINATION_HTTP,
    MCP_INSTALL_PLAN_DESTINATION_STDIO,
    MCP_INSTALL_PLAN_HEADER,
    MCP_INSTALL_PLAN_MISSING_FOOTER,
    MCP_INSTALL_PLAN_NO_CREDENTIALS,
)
from aug.utils.file_settings import McpServerConfig
from aug.utils.hushed import list_secret_names
from aug.utils.mcp_registry import McpRegistryServer
from aug.utils.state import (
    McpCredentialBinding,
    McpInstallPlan,
    load_state,
    save_state,
    update_state,
)

# Turns a target input name (an env var or an HTTP header, e.g. "X-API-Key") into a
# valid hushed secret identifier — the two are never the same thing, since header
# names routinely contain characters hushed secret names can't.
_SECRET_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")


async def get_or_build_plan(
    thread_id: str, index: int, snapshot: list[McpRegistryServer]
) -> McpInstallPlan | None:
    """Return the durable plan for *index* in *thread_id*, building one from
    *snapshot* if this is the first time it's been resolved. None if *index*
    is out of range and there's no persisted plan to fall back on.

    ``list_secret_names`` is awaited before the state read-modify-write below,
    not after: awaiting with state already loaded would let a concurrent
    writer save in between, making this write clobber their change.
    """
    known_secrets = await list_secret_names()
    async with update_state() as state:
        existing = state.mcp.install_plans.get(thread_id)
        if existing is not None and existing.search_index == index:
            refreshed = _revalidate_plan(existing, known_secrets)
            state.mcp.install_plans[thread_id] = refreshed
            return refreshed

        if not (1 <= index <= len(snapshot)):
            return None

        server = snapshot[index - 1]
        plan = build_plan(thread_id, index, server, known_secrets)
        state.mcp.install_plans[thread_id] = plan
        return plan


async def describe_plan(
    thread_id: str, index: int, snapshot: list[McpRegistryServer]
) -> tuple[str, str]:
    """Approval-preview text for installing result *index*: (slug, full plan)."""
    plan = await get_or_build_plan(thread_id, index, snapshot)
    if plan is None:
        return (f"#{index}", "install MCP server")
    return (plan.slug, _render_plan(plan))


def clear_plan(thread_id: str) -> None:
    """Drop the persisted install plan for *thread_id*, if any."""
    state = load_state()
    if state.mcp.install_plans.pop(thread_id, None) is not None:
        save_state(state)


def build_plan(
    thread_id: str, index: int, server: McpRegistryServer, known_secrets: set[str]
) -> McpInstallPlan:
    """Freeze one search result into an immutable install plan."""
    credentials = []
    for name in server.required_inputs:
        secret_name = secret_name_for(name)
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
        literal_inputs=server.literal_inputs,
        created_at=time.time(),
    )


def secret_name_for(target_name: str) -> str:
    """Sanitize an input name (env var or HTTP header) into a hushed secret identifier."""
    name = _SECRET_NAME_SANITIZE_RE.sub("_", target_name).upper()
    return f"_{name}" if not name or name[0].isdigit() else name


def plan_to_config(plan: McpInstallPlan, bindings: dict[str, str]) -> McpServerConfig:
    """Build the settings.json entry for an install plan."""
    if plan.transport == "stdio":
        return McpServerConfig(
            name=plan.slug,
            transport="stdio",
            command=plan.command,
            args=plan.args,
            env=bindings,
            env_static=plan.literal_inputs,
            enabled=True,
        )
    return McpServerConfig(
        name=plan.slug,
        transport="http",
        url=plan.url,
        headers=bindings,
        headers_static=plan.literal_inputs,
        enabled=True,
    )


def render_missing_secrets(plan: McpInstallPlan) -> str:
    """Install-plan preview plus a footer prompting for the still-missing secret(s)."""
    return "\n".join(
        [_render_plan(plan), MCP_INSTALL_PLAN_MISSING_FOOTER.format(index=plan.search_index)]
    )


def _revalidate_plan(plan: McpInstallPlan, known_secrets: set[str]) -> McpInstallPlan:
    refreshed_credentials = [
        c.model_copy(update={"bound": c.secret_name in known_secrets}) for c in plan.credentials
    ]
    if refreshed_credentials == plan.credentials:
        return plan
    return plan.model_copy(update={"credentials": refreshed_credentials})


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

    if plan.literal_inputs:
        lines.append(MCP_INSTALL_PLAN_DEFAULTS_HEADER)
        for target_name, value in plan.literal_inputs.items():
            lines.append(MCP_INSTALL_PLAN_DEFAULT_LINE.format(target_name=target_name, value=value))
    return "\n".join(lines)
