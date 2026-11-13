"""Generic tool-approval mechanism.

Provides:
  - @requires_approval decorator        — when no saved rule matches, asks the user via
                                          ToolContext.request_approval() and waits
  - is_approved / save_approval         — rule engine backed by settings.json
  - list_approvals / revoke_approval    — rule management for /approvals command

Settings schema (tools.approvals):
  [
    {"tool": "run_ssh",           "target": "homeserver", "pattern": "df.*"},
    {"tool": "run_ssh",           "target": "*",          "pattern": "uptime"},
    {"tool": "download_ssh_file", "target": "homeserver", "pattern": ".*"},
    ...
  ]

Rules use re.search (substring match) against the ``operation`` string.
Anchor with ^ / $ for stricter matching.
``target`` and ``tool`` support ``"*"`` as a wildcard that matches anything.
"""

import asyncio
import inspect
import re
from functools import wraps

from aug.core.harness.types import ApprovalDecision, ApprovalRequest
from aug.core.prompts import APPROVAL_NOT_GRANTED_RESULTS
from aug.core.tools.context import current_tool_context
from aug.utils.file_settings import ApprovalRule, load_settings, save_settings

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def requires_approval(fn=None, *, describe=None, on_denied=None):
    """Decorator: ask the user first unless a saved rule approves the operation.

    Apply BEFORE @tool so LangGraph wraps the already-decorated function:

        @tool
        @requires_approval
        async def run_ssh(target: str, command: str) -> str:
            ...

    ``describe`` is a callable (sync or async) that receives the tool's
    kwargs as keyword arguments and returns either:
      - a ``(resource, operation)`` tuple for structured display, or
      - a plain string used as the ``operation`` (resource will be empty).

    Without ``describe``, all kwargs are formatted as ``key: value`` pairs
    and used as the operation.

        @tool
        @requires_approval(describe=lambda target, command: (target, command))
        async def run_ssh(target: str, command: str) -> str:
            ...

    ``on_denied``, when given, is called (sync or async) with the tool's
    kwargs when the operation is not approved — before the result string is
    returned and ``fn`` is skipped. Use it to clean up state the ``describe``
    call may have durably persisted (e.g. an install plan) that would
    otherwise linger as if still pending.

    The tool waits for the user's ApprovalDecision:
      - APPROVED_ONCE   → execute, no rule saved
      - APPROVED_ALWAYS → save exact-match rule, then execute
      - anything else   → run on_denied (if any), return a string saying why the
                          operation was NOT executed
    Outside a turn (no ToolContext) nobody can be asked: the operation is not run.
    """

    def decorator(fn):
        tool_name = fn.__name__

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if describe is not None:
                result = describe(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, tuple):
                    resource, operation = result
                else:
                    resource, operation = "", result
            else:
                resource = ""
                operation = ", ".join(f"{k}: {v}" for k, v in kwargs.items())

            request = ApprovalRequest(tool_name=tool_name, resource=resource, operation=operation)

            if not is_approved(tool_name, resource, operation):
                ctx = current_tool_context()
                try:
                    decision = (
                        await ctx.request_approval(request)
                        if ctx is not None
                        else ApprovalDecision.NO_USER
                    )
                except asyncio.CancelledError:  # /stop while waiting: not approved either
                    await _call(on_denied, kwargs)
                    raise
                if not decision.approved:
                    await _call(on_denied, kwargs)
                    return APPROVAL_NOT_GRANTED_RESULTS[decision.value].format(
                        tool=tool_name, operation=request.description
                    )
                if decision == ApprovalDecision.APPROVED_ALWAYS:
                    save_approval(tool_name, resource, operation)

            return await fn(*args, **kwargs)

        return wrapper

    # Support both @requires_approval and @requires_approval(describe=...)
    if fn is not None:
        return decorator(fn)
    return decorator


def is_approved(tool_name: str, resource: str, operation: str) -> bool:
    """Return True if a saved rule permits *operation* for *tool_name* on *resource*."""
    for rule in load_settings().tools.approvals:
        if rule.tool not in (tool_name, "*"):
            continue
        if rule.target not in (resource, "*"):
            continue
        try:
            if re.search(rule.pattern, operation):
                return True
        except re.error:
            # Corrupted or manually-edited pattern — skip rather than crash.
            continue
    return False


def save_approval(tool_name: str, resource: str, operation: str) -> None:
    """Persist an exact-match approval rule for *operation* on *resource* by *tool_name*.

    No-op if an identical rule already exists.
    """
    s = load_settings()
    pattern = re.escape(operation)
    if any(
        r.tool == tool_name and r.target == resource and r.pattern == pattern
        for r in s.tools.approvals
    ):
        return
    s.tools.approvals.append(ApprovalRule(tool=tool_name, target=resource, pattern=pattern))
    save_settings(s)


def list_approvals() -> list[ApprovalRule]:
    """Return all saved approval rules."""
    return load_settings().tools.approvals


def revoke_approval(index: int) -> None:
    """Remove the approval rule at *index* (0-based).

    Raises:
        IndexError: if *index* is out of range.
    """
    s = load_settings()
    if index < 0 or index >= len(s.tools.approvals):
        n = len(s.tools.approvals)
        raise IndexError(f"Approval index {index} out of range (have {n} rules)")
    s.tools.approvals.pop(index)
    save_settings(s)


async def _call(fn, kwargs: dict) -> None:
    """Call the optional, sync or async *fn* with the tool's kwargs."""
    if fn is None:
        return
    outcome = fn(**kwargs)
    if inspect.isawaitable(outcome):
        await outcome
