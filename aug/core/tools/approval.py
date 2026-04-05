"""Generic tool-approval mechanism.

Provides:
  - ApprovalRequest / ApprovalDecision  — shared types
  - @requires_approval decorator        — pauses via LangGraph interrupt() when
                                          no saved rule matches; resumes with the
                                          user's decision
  - is_approved / save_approval         — rule engine backed by settings.json
  - list_approvals / revoke_approval    — rule management for /approvals command

Settings schema (approvals):
  [
    {"pattern": "homeserver: df.*"},
    {"pattern": "homeserver: systemctl status .*"},
    ...
  ]

Rules use re.search (substring match) against ``resource: operation``.
Anchor with ^ / $ for stricter matching.
"""

import re
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from langgraph.types import interrupt

from aug.utils.user_settings import get_setting, set_setting

_SETTING_PATH = ("approvals",)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalRequest:
    """Payload passed to interrupt() when an operation needs user approval.

    Attributes:
        resource:  What is being acted on (e.g. an SSH target, a file path).
                   Empty string if the tool has no single resource concept.
        operation: What is being done (e.g. a shell command, a file transfer).

    Rule matching uses the combined ``"resource: operation"`` string (or just
    ``operation`` when resource is empty).
    """

    resource: str
    operation: str

    @property
    def description(self) -> str:
        """Combined string used for rule matching and denial messages."""
        return f"{self.resource}: {self.operation}" if self.resource else self.operation


class ApprovalDecision(Enum):
    APPROVED_ONCE = "approved_once"
    APPROVED_ALWAYS = "approved_always"
    DENIED = "denied"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def requires_approval(fn=None, *, describe=None):
    """Decorator: pause via LangGraph interrupt() if no saved rule approves the operation.

    Apply BEFORE @tool so LangGraph wraps the already-decorated function:

        @tool
        @requires_approval
        async def run_ssh(target: str, command: str) -> str:
            ...

    ``describe`` is a callable that receives the tool's kwargs as keyword
    arguments and returns either:
      - a ``(resource, operation)`` tuple for structured display, or
      - a plain string used as the ``operation`` (resource will be empty).

    Without ``describe``, all kwargs are formatted as ``key: value`` pairs
    and used as the operation.

        @tool
        @requires_approval(describe=lambda target, command: (target, command))
        async def run_ssh(target: str, command: str) -> str:
            ...

    On resume the decorator acts on the user's ApprovalDecision:
      - APPROVED_ONCE   → execute immediately, no rule saved
      - APPROVED_ALWAYS → save exact-match rule, then execute
      - DENIED          → return a clear denial string, do NOT execute
    """

    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if describe is not None:
                result = describe(**kwargs)
                if isinstance(result, tuple):
                    resource, operation = result
                else:
                    resource, operation = "", result
            else:
                resource = ""
                operation = ", ".join(f"{k}: {v}" for k, v in kwargs.items())

            request = ApprovalRequest(resource=resource, operation=operation)

            if not is_approved(request.description):
                decision: ApprovalDecision = interrupt(request)
                if decision == ApprovalDecision.DENIED:
                    return f"Operation denied by user. '{request.description}' was NOT executed."
                if decision == ApprovalDecision.APPROVED_ALWAYS:
                    save_approval(request.description)

            return await fn(*args, **kwargs)

        return wrapper

    # Support both @requires_approval and @requires_approval(describe=...)
    if fn is not None:
        return decorator(fn)
    return decorator


def is_approved(description: str) -> bool:
    """Return True if a saved rule permits *description* (``resource: operation``)."""
    rules: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    for rule in rules:
        pattern = rule.get("pattern", "")
        try:
            if re.search(pattern, description):
                return True
        except re.error:
            # Corrupted or manually-edited pattern — skip rather than crash.
            continue
    return False


def save_approval(description: str) -> None:
    """Persist an exact-match approval rule for *description*.

    No-op if an identical rule already exists.
    """
    rules: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    pattern = re.escape(description)
    if any(r.get("pattern") == pattern for r in rules):
        return
    rules.append({"pattern": pattern})
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
