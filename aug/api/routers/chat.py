"""Chat router — /chat/invoke and /chat/stream."""

import json
import logging
from uuid import uuid4

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from aug.api.schemas.chat import ChatRequest, ChatResponse
from aug.api.security import require_api_key
from aug.core.events import ChatModelStreamEvent
from aug.core.registry import get_agent, list_agents
from aug.core.run import run_registry
from aug.core.state import AgentState
from aug.utils.logging import set_correlation_id, set_thread_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)


def _get_checkpointer(request: Request):
    """Pull the shared checkpointer from app state."""
    return request.app.state.checkpointer


def _validate_agent(agent: str) -> None:
    if agent not in list_agents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent '{agent}'. Available: {list_agents()}",
        )


@router.post("/invoke", response_model=ChatResponse)
async def invoke(body: ChatRequest, request: Request) -> ChatResponse:
    """Run the agent and return the full response as JSON."""
    set_correlation_id(str(uuid4())[:8])
    set_thread_id(body.thread_id)
    _validate_agent(body.agent)
    checkpointer = _get_checkpointer(request)
    agent = get_agent(body.agent)

    state = AgentState(
        messages=[HumanMessage(content=body.message)],
        thread_id=body.thread_id,
    )
    config: RunnableConfig = {"configurable": {"thread_id": body.thread_id}}

    logger.info("invoke thread=%s agent=%s", body.thread_id, body.agent)
    text = ""
    try:
        async for event in agent.astream_events(state, config, checkpointer):
            if isinstance(event, ChatModelStreamEvent) and event.delta:
                text += event.delta
    except psycopg.OperationalError as exc:
        logger.exception("DB connection lost during invoke")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection lost. Please retry.",
        ) from exc

    return ChatResponse(
        thread_id=body.thread_id,
        agent=body.agent,
        response=text,
        tool_calls=[],
    )


@router.post("/stream")
async def stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the agent and stream the response as Server-Sent Events."""
    set_correlation_id(str(uuid4())[:8])
    set_thread_id(body.thread_id)
    _validate_agent(body.agent)
    checkpointer = _get_checkpointer(request)
    agent = get_agent(body.agent)

    state = AgentState(
        messages=[HumanMessage(content=body.message)],
        thread_id=body.thread_id,
    )
    config: RunnableConfig = {"configurable": {"thread_id": body.thread_id}}

    async def event_generator():
        logger.info("stream thread=%s agent=%s", body.thread_id, body.agent)
        try:
            async for event in agent.astream_events(state, config, checkpointer):
                match event:
                    case ChatModelStreamEvent(delta=delta) if delta:
                        yield f"event: text_delta\ndata: {json.dumps({'delta': delta})}\n\n"
        except psycopg.OperationalError:
            logger.exception("DB connection lost during stream")
            detail = json.dumps({"detail": "Database connection lost. Please retry."})
            yield f"event: error\ndata: {detail}\n\n"
            return

        done_payload = json.dumps({"thread_id": body.thread_id, "agent": body.agent})
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
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
