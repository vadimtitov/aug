"""Command approval mechanism for tools that execute on remote systems.

Provides:
  - ApprovalRequest / ApprovalDecision  — shared types
  - @requires_approval decorator        — pauses via LangGraph interrupt() when
                                          no saved rule matches; resumes with the
                                          user's decision
  - is_approved / save_approval         — rule engine backed by settings.json
  - list_approvals / revoke_approval    — rule management for /approvals command

Settings schema:
  tools.ssh.approvals = [
    {"target": "homeserver", "pattern": "df.*"},
    ...
  ]

Rules use re.search (substring match). Anchor with ^ / $ for stricter matching.
Target "*" matches any target.
"""

import re
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from langgraph.types import interrupt

from aug.utils.user_settings import get_setting, set_setting

_SETTING_PATH = ("tools", "ssh", "approvals")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalRequest:
    """Payload passed to interrupt() when a command needs user approval."""

    target: str
    command: str


class ApprovalDecision(Enum):
    APPROVED_ONCE = "approved_once"
    APPROVED_ALWAYS = "approved_always"
    DENIED = "denied"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def requires_approval(fn):
    """Decorator: pause via LangGraph interrupt() if no saved rule approves the command.

    Apply BEFORE @tool so LangGraph wraps the already-decorated function:

        @tool
        @requires_approval
        async def run_ssh(target: str, command: str) -> str:
            ...

    The decorated function must accept ``target`` and ``command`` as keyword
    arguments.  On resume the decorator acts on the user's ApprovalDecision:
      - APPROVED_ONCE   → execute immediately, no rule saved
      - APPROVED_ALWAYS → save exact-match rule, then execute
      - DENIED          → return a clear denial string, do NOT execute
    """

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        target: str = kwargs.get("target", "")
        command: str = kwargs.get("command", "")

        if not is_approved(target, command):
            decision: ApprovalDecision = interrupt(ApprovalRequest(target=target, command=command))
            if decision == ApprovalDecision.DENIED:
                return (
                    f"Command denied by user. "
                    f"The command '{command}' on '{target}' was NOT executed."
                )
            if decision == ApprovalDecision.APPROVED_ALWAYS:
                save_approval(target, command)

        return await fn(*args, **kwargs)

    return wrapper


def is_approved(target: str, command: str) -> bool:
    """Return True if a saved rule permits *command* on *target*."""
    rules: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    for rule in rules:
        rule_target = rule.get("target", "")
        pattern = rule.get("pattern", "")
        if rule_target not in (target, "*"):
            continue
        if re.search(pattern, command):
            return True
    return False


def save_approval(target: str, command: str) -> None:
    """Persist an exact-match approval rule for *command* on *target*."""
    rules: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    rules.append({"target": target, "pattern": re.escape(command)})
    set_setting(*_SETTING_PATH, value=rules)


def list_approvals() -> list[dict]:
    """Return all saved approval rules."""
    return get_setting(*_SETTING_PATH, default=[]) or []


def revoke_approval(index: int) -> None:
    """Remove the approval rule at *index* (0-based).

    Raises:
        IndexError: if *index* is out of range.
    """
    rules: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    if index < 0 or index >= len(rules):
        raise IndexError(f"Approval index {index} out of range (have {len(rules)} rules)")
    rules.pop(index)
    set_setting(*_SETTING_PATH, value=rules)
