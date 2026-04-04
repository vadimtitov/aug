"""Chat router — /chat/invoke and /chat/stream."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from aug.api.interfaces.rest import RestApiInterface
from aug.api.schemas.chat import ApprovalRequest, ChatRequest, ChatResponse
from aug.api.security import require_api_key
from aug.core.registry import list_agents
from aug.core.run import run_registry
from aug.core.tools.approval import ApprovalDecision
from aug.utils.logging import set_thread_id

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
    set_thread_id(body.thread_id)
    _validate_agent(body.agent)
    interface = _get_interface(request)
    logger.info("invoke thread=%s agent=%s", body.thread_id, body.agent)
    text = await interface.invoke(body)
    return ChatResponse(
        thread_id=body.thread_id,
        agent=body.agent,
        response=text,
        tool_calls=[],
    )


@router.post("/stream")
async def stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the agent and stream the response as Server-Sent Events."""
    set_thread_id(body.thread_id)
    _validate_agent(body.agent)
    interface = _get_interface(request)
    return StreamingResponse(
        interface.stream_sse(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{thread_id}/run/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(thread_id: str) -> None:
    """Cancel the currently active agent run for a thread, if any."""
    run = run_registry.get(thread_id)
    if run and run.active:
        logger.info("cancel_run thread=%s run=%s", thread_id, run.id)
        run.request_stop()


@router.post("/{thread_id}/approve", response_model=ChatResponse)
async def approve_command(thread_id: str, body: ApprovalRequest, request: Request) -> ChatResponse:
    """Resume an agent run paused on a command approval interrupt.

    Call this after receiving the approval prompt text in the /invoke or /stream
    response.  The agent will continue with the given decision and return its
    next response.
    """
    set_thread_id(thread_id)
    _validate_agent(body.agent)
    decision = ApprovalDecision(body.decision)
    sender_id = body.sender_id or thread_id
    interface = _get_interface(request)
    if await interface.get_pending_approval(thread_id, body.agent) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Thread '{thread_id}' has no pending approval interrupt.",
        )
    logger.info("approve_command thread=%s agent=%s decision=%s", thread_id, body.agent, decision)
    text = await interface.invoke_resume(thread_id, body.agent, sender_id, decision)
    return ChatResponse(thread_id=thread_id, agent=body.agent, response=text, tool_calls=[])


def _get_interface(request: Request) -> RestApiInterface:
    return RestApiInterface(request.app.state.checkpointer)
