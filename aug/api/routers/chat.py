"""Chat router — /chat/invoke, /chat/stream and approvals."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from aug.api.interfaces.rest import RestApiInterface
from aug.api.schemas.chat import ApprovalAnswer, ChatRequest, ChatResponse, PendingApprovalInfo
from aug.api.security import require_api_key
from aug.core.harness import ApprovalDecision, pending_approvals
from aug.core.registry import list_agents
from aug.utils.logging import set_context_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)


def _validate_agent(agent: str) -> None:
    if agent not in list_agents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent '{agent}'. Available: {list_agents()}",
        )


@router.post("/invoke", response_model=ChatResponse)
async def invoke(body: ChatRequest, request: Request) -> ChatResponse:
    """Run the agent and return the full response as JSON."""
    set_context_id(body.context_id)
    interface = _get_interface(request)
    agent = body.agent or interface.get_agent_version(body.context_id)
    _validate_agent(agent)
    logger.info("invoke thread=%s agent=%s", body.context_id, agent)
    text = await interface.invoke(body, agent)
    return ChatResponse(
        context_id=body.context_id,
        agent=agent,
        response=text,
        tool_calls=[],
    )


@router.post("/stream")
async def stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the agent and stream the response as Server-Sent Events."""
    set_context_id(body.context_id)
    interface = _get_interface(request)
    agent = body.agent or interface.get_agent_version(body.context_id)
    _validate_agent(agent)
    return StreamingResponse(
        interface.stream_sse(body, agent),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{context_id}/run/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(context_id: str, request: Request) -> None:
    """Cancel the currently active agent run for a thread, if any."""
    if await _get_interface(request).stop_run(context_id):
        logger.info("cancel_run thread=%s", context_id)


@router.get("/{context_id}/approvals", response_model=list[PendingApprovalInfo])
async def list_pending_approvals(context_id: str) -> list[PendingApprovalInfo]:
    """Approval requests a running turn in *context_id* is waiting on.

    The same requests arrive as ``approval_requested`` events on /chat/stream; this is
    for /chat/invoke callers, whose response only returns once the turn is over.
    """
    return [
        PendingApprovalInfo(
            id=p.id,
            tool_name=p.request.tool_name,
            resource=p.request.resource,
            operation=p.request.operation,
        )
        for p in pending_approvals.for_context(context_id)
    ]


@router.post("/{context_id}/approve", status_code=status.HTTP_204_NO_CONTENT)
async def approve(context_id: str, body: ApprovalAnswer) -> None:
    """Answer an approval request.  The waiting turn carries on in its own response."""
    set_context_id(context_id)
    pending = pending_approvals.get(body.id)
    decision = ApprovalDecision(body.decision)
    if (
        pending is None
        or pending.context_id != context_id
        or not pending_approvals.resolve(body.id, decision)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Approval '{body.id}' is not pending in thread '{context_id}'.",
        )
    logger.info("approve thread=%s id=%s decision=%s", context_id, body.id, decision.value)


def _get_interface(request: Request) -> RestApiInterface:
    return RestApiInterface()
