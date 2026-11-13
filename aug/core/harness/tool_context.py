"""What a tool gets from the harness running it: who it runs for, and a way to talk back.

A tool reports progress with ``ctx.emit()``, sends the user a file with
``ctx.send_file()`` and asks for approval with ``await ctx.request_approval()``; the
approval is awaited inside the tool, so the turn never pauses.  Every harness builds a
ToolContext per turn and hands it to its tools.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from aug.core.harness.types import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
    FileEmitted,
    TurnContext,
)
from aug.utils.storage import StoredFile

logger = logging.getLogger(__name__)

APPROVAL_TIMEOUT_SECONDS = 30 * 60

# Where a LangGraph run carries the ToolContext in its config's configurable dict.
TOOL_CONTEXT_CONFIG_KEY = "tool_context"


@dataclass(frozen=True)
class PendingApproval:
    id: str
    context_id: str
    request: ApprovalRequest


class PendingApprovals:
    """Approval requests waiting for the user, keyed by request id.

    In memory: a request pending at restart is lost with the turn that asked it.
    Use the module-level ``pending_approvals`` singleton.
    """

    def __init__(self) -> None:
        self._waiting: dict[str, tuple[PendingApproval, asyncio.Future[ApprovalDecision]]] = {}

    def open(
        self, context_id: str, request: ApprovalRequest
    ) -> tuple[PendingApproval, asyncio.Future[ApprovalDecision]]:
        """Register *request*; the future resolves with the user's decision.  It may be
        answered as soon as this returns, before anyone has shown it."""
        pending = PendingApproval(id=uuid4().hex[:12], context_id=context_id, request=request)
        future = asyncio.get_running_loop().create_future()
        self._waiting[pending.id] = (pending, future)
        return pending, future

    def close(self, approval_id: str) -> None:
        """Stop waiting for *approval_id*: a later tap on it is "already handled"."""
        entry = self._waiting.pop(approval_id, None)
        if entry is not None and not entry[1].done():
            entry[1].cancel()

    def get(self, approval_id: str) -> PendingApproval | None:
        entry = self._waiting.get(approval_id)
        return entry[0] if entry else None

    def resolve(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Answer *approval_id*.  False if it is no longer waiting (already answered,
        timed out, stopped), so a second tap never lands on another request."""
        entry = self._waiting.pop(approval_id, None)
        if entry is None or entry[1].done():
            return False
        entry[1].set_result(decision)
        return True

    def for_context(self, context_id: str) -> list[PendingApproval]:
        return [p for p, _ in self._waiting.values() if p.context_id == context_id]

    def deny_all(self, context_id: str, decision: ApprovalDecision) -> int:
        """Answer every request waiting in *context_id* with *decision*; how many."""
        return sum(self.resolve(p.id, decision) for p in self.for_context(context_id))

    def clear(self) -> None:
        self._waiting.clear()


pending_approvals = PendingApprovals()


@dataclass
class ToolContext:
    """One turn's context, shared by every tool call in it.

    context_id: the session the turn runs in.
    stopped:    set by cancel(); slow tools poll it to stop early.
    """

    context_id: str
    turn: TurnContext
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    _emit: Callable[[AgentEvent], Awaitable[None]] | None = None

    async def emit(self, event: AgentEvent) -> None:
        """Add *event* to the turn's stream, under the tool call that emits it."""
        if self._emit is not None:
            await self._emit(event)

    async def send_file(self, stored: StoredFile, caption: str = "") -> None:
        """Send *stored* (already in the outbox) to the user, under the calling tool."""
        await self.emit(
            FileEmitted(
                file_id=stored.id,
                path=stored.path,
                media_type=stored.media_type,
                filename=stored.filename,
                caption=caption,
            )
        )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask the user to approve *request* and wait for the answer.

        An unattended turn has nobody to ask and gets NO_USER at once.  A wait that
        outlasts APPROVAL_TIMEOUT_SECONDS gets TIMED_OUT.
        """
        if not self.turn.attended:
            return ApprovalDecision.NO_USER
        pending, answer = pending_approvals.open(self.context_id, request)
        logger.info(
            "approval_requested context=%s id=%s tool=%s",
            self.context_id,
            pending.id,
            request.tool_name,
        )
        try:
            await self.emit(ApprovalRequested(id=pending.id, request=request))
            try:
                decision = await asyncio.wait_for(answer, APPROVAL_TIMEOUT_SECONDS)
            except TimeoutError:
                decision = ApprovalDecision.TIMED_OUT
        finally:
            # However the wait ended, /stop (cancellation) included.
            pending_approvals.close(pending.id)
        logger.info(
            "approval_resolved context=%s id=%s decision=%s",
            self.context_id,
            pending.id,
            decision.value,
        )
        await self.emit(ApprovalResolved(id=pending.id, decision=decision))
        return decision
