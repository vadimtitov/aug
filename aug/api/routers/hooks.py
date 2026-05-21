"""Webhook endpoint for external systems to push messages to the agent.

POST /hooks/push

Supports two delivery types:

  type="forward"
      The message is relayed to the target thread as plain text — no agent
      processing.  Returns 200 when delivery is confirmed.

  type="agent"
      An agent run is launched in a background task with a constrained
      recursion limit.  Returns 202 immediately; the result is delivered to
      the target thread when the run completes.

Authentication: standard X-API-Key / Bearer JWT (same as all other routes).

Thread routing:
  thread_id="default"  → each interface resolves its own default target
                         (Telegram: the DM with the bot).
  thread_id="new"      → create a new thread (Telegram: forum topic).
                         Requires chat_id and optional topic_name in payload.
  thread_id="tg-…"     → deliver to that specific thread.
"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aug.api.security import require_api_key
from aug.core.dispatch import InterfaceName, fire_push

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hooks", tags=["hooks"])


class PushRequest(BaseModel):
    """Payload for POST /hooks/push."""

    interface: InterfaceName
    """Interface name, e.g. ``"telegram"``."""

    thread_id: str = "default"
    """Target thread.  ``"default"`` resolves per-interface, ``"new"`` creates a thread."""

    message: str
    """Text to forward or use as the agent prompt."""

    type: Literal["forward", "agent", "agent_isolated", "inject"] = "forward"
    """Delivery mode.
    ``"forward"``        — relay message as-is, no agent run.
    ``"agent"``          — agent run in the thread's existing history.
    ``"agent_isolated"`` — agent run in a throwaway session (no history shared).
    ``"inject"``         — write message to context + forward to chat, no reply.
    """

    topic_name: str | None = None
    """Topic name when ``thread_id="new"`` and the interface supports topic creation."""

    chat_id: int | None = None
    """Group chat ID when ``thread_id="new"`` (required for Telegram forum topics)."""


@router.post("/push", status_code=200)
async def push(
    payload: PushRequest,
    request: Request,
    _: None = Depends(require_api_key),
):
    """Deliver a message to a target thread via the specified interface.

    Returns 200 for ``type="forward"`` (synchronous delivery) and 202 for
    ``type="agent"`` (async, fire-and-forget).
    """
    interfaces = getattr(request.app.state, "interfaces", {})
    if payload.interface not in interfaces:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interface '{payload.interface}' is not registered or not running.",
        )

    if payload.type in ("forward", "inject"):
        await fire_push(
            request.app,
            interface=payload.interface,
            thread_id=payload.thread_id,
            message=payload.message,
            push_type=payload.type,
            topic_name=payload.topic_name,
            chat_id=payload.chat_id,
        )
        return {"status": "delivered"}

    # type == "agent" or "agent_isolated" — fire and forget
    asyncio.create_task(  # noqa: RUF006 — intentional fire-and-forget
        _guarded_push(request.app, payload),
        name=f"push-agent-{payload.interface}",
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "queued"})


async def _guarded_push(app, payload: PushRequest) -> None:
    """Fire fire_push and log any exception rather than silently dropping it."""
    try:
        await fire_push(
            app,
            interface=payload.interface,
            thread_id=payload.thread_id,
            message=payload.message,
            push_type=payload.type,
            topic_name=payload.topic_name,
            chat_id=payload.chat_id,
        )
    except Exception:
        logger.exception(
            "push-agent failed interface=%s thread_id=%s",
            payload.interface,
            payload.thread_id,
        )
