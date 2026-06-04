"""Browser view router — live screencast of the agent-controlled browser.

  GET  /browser/status   → whether the live browser view is available
  WS   /browser/stream   → binary JPEG frames of the tab the agent is on

The WebSocket credential travels in the ``Sec-WebSocket-Protocol`` header (the one
header a browser WebSocket client can set), not the URL — keeping it out of access
logs and history. The client offers two subprotocols: a fixed marker and its JWT.
Frames are sent as binary messages (raw JPEG); the client paints the latest one.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from aug.api.security import require_api_key, verify_ws_credential
from aug.core.browser_view import BrowserViewHub, Viewer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/browser", tags=["browser"])

# Fixed subprotocol the client offers alongside its JWT; we echo it on accept.
_SUBPROTOCOL = "aug.browser-view.v1"

# Resend the last frame if nothing new arrives for this long, so the connection
# stays warm through idle proxies while the agent is "thinking" between actions.
_KEEPALIVE_SECONDS = 20.0

# Custom WebSocket close codes (4000-4999 is the application-private range).
_CLOSE_UNAUTHORIZED = 4401
_CLOSE_UNAVAILABLE = 4404


@router.get("/status", dependencies=[Depends(require_api_key)])
async def status(request: Request) -> dict[str, bool]:
    """Report whether a live browser view can be opened."""
    hub: BrowserViewHub = request.app.state.browser_view_hub
    return {"available": hub.configured}


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    # The credential is the offered subprotocol that isn't our marker.
    offered = websocket.scope.get("subprotocols", [])
    token = next((p for p in offered if p != _SUBPROTOCOL), "")
    # Accept first (echoing the marker) so our application close codes reach the
    # browser — rejecting before the handshake completes surfaces only opaque 1006.
    await websocket.accept(subprotocol=_SUBPROTOCOL if _SUBPROTOCOL in offered else None)
    if not verify_ws_credential(token):
        await websocket.close(code=_CLOSE_UNAUTHORIZED)
        return
    hub: BrowserViewHub = websocket.app.state.browser_view_hub
    if not hub.configured:
        await websocket.close(code=_CLOSE_UNAVAILABLE)
        return

    try:
        async with hub.view() as viewer:
            sender = asyncio.create_task(_send_frames(websocket, viewer))
            receiver = asyncio.create_task(_drain(websocket))
            done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            # Reap both sides so neither is GC'd with an unretrieved exception (a
            # disconnect surfaces as WebSocketDisconnect on the drain task; the
            # cancelled side surfaces as CancelledError). Both are expected here.
            for task in (*done, *pending):
                try:
                    await task
                except (asyncio.CancelledError, WebSocketDisconnect):
                    pass
                except Exception:
                    logger.debug("browser stream task error", exc_info=True)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("browser stream error")


async def _send_frames(websocket: WebSocket, viewer: Viewer) -> None:
    """Forward the latest frame to the client; resend on idle to keep alive."""
    last: bytes | None = None
    while True:
        try:
            frame = await asyncio.wait_for(viewer.get(), timeout=_KEEPALIVE_SECONDS)
        except TimeoutError:
            if last is not None:
                await websocket.send_bytes(last)
            continue
        if frame is None:  # viewer closed (hub shutting down)
            return
        last = frame
        await websocket.send_bytes(frame)


async def _drain(websocket: WebSocket) -> None:
    """Consume client messages so a disconnect is detected promptly."""
    while True:
        await websocket.receive()
