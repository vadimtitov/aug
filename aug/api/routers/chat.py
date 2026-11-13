"""Chat router — /chat/invoke and /chat/stream."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from aug.api.interfaces.rest import RestApiInterface
from aug.api.schemas.chat import ApprovalRequest, ChatRequest, ChatResponse
from aug.api.security import require_api_key
from aug.core.registry import list_agents
from aug.core.tools.approval import ApprovalDecision
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


@router.post("/{context_id}/approve", response_model=ChatResponse)
async def approve_command(context_id: str, body: ApprovalRequest, request: Request) -> ChatResponse:
    """Resume an agent run paused on a command approval interrupt.

    Call this after receiving the approval prompt text in the /invoke or /stream
    response.  The agent will continue with the given decision and return its
    next response.
    """
    set_context_id(context_id)
    interface = _get_interface(request)
    agent = body.agent or interface.get_agent_version(context_id)
    _validate_agent(agent)
    decision = ApprovalDecision(body.decision)
    sender_id = body.sender_id or context_id
    if await interface.get_pending_approval(context_id, agent) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Thread '{context_id}' has no pending approval interrupt.",
        )
    logger.info("approve_command thread=%s agent=%s decision=%s", context_id, agent, decision)
    text = await interface.invoke_resume(context_id, agent, sender_id, decision)
    return ChatResponse(context_id=context_id, agent=agent, response=text, tool_calls=[])


def _get_interface(request: Request) -> RestApiInterface:
    return RestApiInterface()
