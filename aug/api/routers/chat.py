"""Chat router — /chat/invoke and /chat/stream."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from aug.api.schemas.chat import ChatRequest, ChatResponse
from aug.api.security import require_api_key
from aug.core.registry import get_agent, list_agents

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
    _validate_agent(body.agent)
    checkpointer = _get_checkpointer(request)
    graph = get_agent(body.agent, checkpointer)

    config = {"configurable": {"thread_id": body.thread_id}}
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "thread_id": body.thread_id,
    }

    logger.info("invoke thread=%s agent=%s", body.thread_id, body.agent)
    result = await graph.ainvoke(input_state, config=config)

    last_ai = next(
        (m for m in reversed(result["messages"]) if m.type == "ai"),
        None,
    )
    response_text = last_ai.content if last_ai else ""

    return ChatResponse(
        thread_id=body.thread_id,
        agent=body.agent,
        response=response_text,
        tool_calls=[],
    )


@router.post("/stream")
async def stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the agent and stream the response as Server-Sent Events."""
    _validate_agent(body.agent)
    checkpointer = _get_checkpointer(request)
    graph = get_agent(body.agent, checkpointer)

    config = {"configurable": {"thread_id": body.thread_id}}
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "thread_id": body.thread_id,
    }

    async def event_generator():
        logger.info("stream thread=%s agent=%s", body.thread_id, body.agent)

        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                delta = event["data"]["chunk"].content
                if delta:
                    yield f"event: text_delta\ndata: {json.dumps({'delta': delta})}\n\n"

            elif kind == "on_tool_start":
                payload = {"name": event["name"], "input": event["data"].get("input", {})}
                yield f"event: tool_call\ndata: {json.dumps(payload)}\n\n"

            elif kind == "on_tool_end":
                payload = {"name": event["name"], "output": event["data"].get("output", {})}
                yield f"event: tool_result\ndata: {json.dumps(payload)}\n\n"

        done_payload = json.dumps({"thread_id": body.thread_id, "agent": body.agent})
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
