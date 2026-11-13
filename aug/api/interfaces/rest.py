"""REST API interface — adapts HTTP chat requests into the BaseInterface pipeline."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from aug.api.interfaces.base import (
    BaseInterface,
    ContentPart,
    FileContent,
    IncomingMessage,
    TextContent,
)
from aug.api.schemas.chat import ChatFile, ChatRequest
from aug.core.harness import (
    ApprovalRequested,
    ApprovalResolved,
    FileEmitted,
    HarnessEvent,
    StatusUpdate,
    TextDelta,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnEnded,
    TurnStarted,
)
from aug.utils.storage import file_store

logger = logging.getLogger(__name__)

# One SSE event: its name and its JSON payload.
Frame = tuple[str, dict[str, Any]]

# An SSE comment sent when nothing else was, so proxies keep an idle stream open (a
# tool waiting for approval can be silent for minutes).
_KEEPALIVE_SECONDS = 15.0


@dataclass
class _RestContext:
    """Holds the incoming request and a queue that receives the response's frames."""

    request: ChatRequest
    agent_version: str
    _queue: asyncio.Queue[Frame | None] = field(default_factory=asyncio.Queue, init=False)

    def put(self, name: str, payload: dict[str, Any]) -> None:
        self._queue.put_nowait((name, payload))

    def close(self) -> None:
        """Signal end-of-response to any consumer of this context."""
        self._queue.put_nowait(None)

    async def next_frame(self) -> Frame | None:
        """The next frame; None at end of response.  Safe to cancel."""
        return await self._queue.get()

    async def collect(self) -> tuple[str, list[ChatFile]]:
        """Drain the queue into the reply text and the files sent (used by /invoke)."""
        parts: list[str] = []
        files: list[ChatFile] = []
        async for name, payload in self:
            if name in ("text_delta", "message"):
                parts.append(payload["text"])
            elif name == "file":
                files.append(ChatFile(**payload))
        return "".join(parts), files

    def __aiter__(self) -> AsyncIterator[Frame]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Frame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame


class RestApiInterface(BaseInterface[_RestContext]):
    """REST API frontend — runs every request through the full BaseInterface pipeline.

    Both /chat/invoke and /chat/stream benefit from reflexes, structured logging,
    run lifecycle management, and error handling defined in BaseInterface.
    """

    async def receive_message(self, context: _RestContext) -> IncomingMessage | None:
        req = context.request
        parts: list[ContentPart] = []
        for attachment in req.attachments:
            # Checked by the router; a file gone since then is left out.
            stored = file_store.get(attachment.file_id)
            if stored is not None:
                parts.append(
                    FileContent(
                        path=str(stored.path), mime_type=stored.media_type, filename=stored.filename
                    )
                )
        if req.message:
            parts.append(TextContent(text=req.message))
        return IncomingMessage(
            parts=parts,
            interface="rest_api",
            sender_id=req.context_id,
            context_id=req.context_id,
            agent_version=context.agent_version,
            user_id=req.context_id,  # REST has no separate user identity — the thread is the caller
        )

    async def send_stream(self, stream: AsyncIterator[HarnessEvent], context: _RestContext) -> None:
        """One SSE frame per harness event.  An approval_requested frame is answered
        with POST /chat/{context_id}/approve while the stream stays open."""
        async for event in stream:
            frame = _frame(event)
            if frame is not None:
                context.put(*frame)

    async def send_message(self, message: str, context: _RestContext) -> None:
        context.put("message", {"text": message})

    async def send_notification(self, target_id: str, text: str) -> None:
        pass  # REST has no push channel — reminders set during REST sessions are silently dropped

    async def resolve_context(
        self,
        context_id: str,
        *,
        topic_name: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """Return *context_id* unchanged; raise for logical aliases unsupported by REST.

        REST has no persistent connection, so ``"default"`` and ``"new"`` have no
        meaning.  Only concrete thread IDs are valid here.
        """
        if context_id in ("default", "new"):
            raise ValueError(
                f"REST API does not support {context_id!r} thread resolution — "
                "push delivery requires a Telegram interface."
            )
        return context_id

    def conversation_id(self, context_id: str) -> str:
        """REST thread IDs are stable and client-chosen, so each is its own conversation."""
        return f"rest-{context_id}"

    async def send_proactive(self, context_id: str, text: str) -> None:
        """No-op: REST has no push channel for forward-type pushes."""

    async def announcement_threads(self) -> list[str]:
        """No targets: REST has no push channel, so announcements have nowhere to land."""
        return []

    async def send_proactive_stream(
        self, context_id: str, stream: AsyncIterator[HarnessEvent]
    ) -> None:
        """Consume and discard the stream; REST has no push channel."""
        async for _ in stream:
            pass

    async def invoke(self, request: ChatRequest, agent_version: str) -> tuple[str, list[ChatFile]]:
        """Run the full pipeline and return the complete response text and its files."""
        ctx = _RestContext(request=request, agent_version=agent_version)
        task = asyncio.create_task(self._run_and_close(ctx))
        reply = await ctx.collect()
        await task
        return reply

    async def stream_sse(self, request: ChatRequest, agent_version: str) -> AsyncIterator[str]:
        """Run the full pipeline, yielding Server-Sent Events."""
        ctx = _RestContext(request=request, agent_version=agent_version)
        task = asyncio.create_task(self._run_and_close(ctx))
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(ctx.next_frame(), _KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if frame is None:
                    break
                yield _sse(*frame)
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        yield _sse("done", {"context_id": request.context_id, "agent": agent_version})

    async def _run_and_close(self, ctx: _RestContext) -> None:
        try:
            await self.run(ctx)
        finally:
            ctx.close()


def _frame(event: HarnessEvent) -> Frame | None:
    match event:
        case TurnStarted(context_id=context_id):
            return "turn_started", {"context_id": context_id}
        case TextDelta(text=text, parent_id=parent_id):
            return ("text_delta", {"text": text}) if text and parent_id is None else None
        case ToolStarted(id=tool_id, name=name, args=args, parent_id=parent_id):
            return "tool_started", {
                "id": tool_id,
                "name": name,
                "args": args,
                "parent_id": parent_id,
            }
        case ToolProgress(text=text, parent_id=parent_id):
            return "tool_progress", {"text": text, "parent_id": parent_id}
        case ToolEnded(id=tool_id, name=name, error=error, parent_id=parent_id):
            return "tool_ended", {
                "id": tool_id,
                "name": name,
                "error": error,
                "parent_id": parent_id,
            }
        case FileEmitted(
            file_id=file_id,
            filename=filename,
            media_type=media_type,
            caption=caption,
            parent_id=parent_id,
        ):
            return "file", {
                "id": file_id,
                "filename": filename,
                "media_type": media_type,
                "caption": caption,
                "url": f"/files/{file_id}/content",
                "parent_id": parent_id,
            }
        case StatusUpdate(text=text):
            return "status", {"text": text}
        case ApprovalRequested(id=approval_id, request=request, parent_id=parent_id):
            return "approval_requested", {
                "id": approval_id,
                "tool_name": request.tool_name,
                "resource": request.resource,
                "operation": request.operation,
                "parent_id": parent_id,
            }
        case ApprovalResolved(id=approval_id, decision=decision, parent_id=parent_id):
            return "approval_resolved", {
                "id": approval_id,
                "decision": decision.value,
                "parent_id": parent_id,
            }
        case TurnEnded(error=error, cancelled=cancelled):
            return "turn_ended", {
                "error": error.kind if error else None,
                "cancelled": cancelled,
            }
    return None


def _sse(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"
