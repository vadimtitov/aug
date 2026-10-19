"""REST API interface — adapts HTTP chat requests into the BaseInterface pipeline."""

import asyncio
import contextlib
import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from aug.api.interfaces.base import (
    BaseInterface,
    ContentPart,
    FileContent,
    IncomingMessage,
    TextContent,
)
from aug.api.schemas.chat import ChatRequest
from aug.core.events import AgentEvent, ChatModelStreamEvent
from aug.core.tools.approval import ApprovalDecision, ApprovalRequest
from aug.utils.data import UPLOADS_DIR
from aug.utils.storage import FileStorage

logger = logging.getLogger(__name__)


@dataclass
class _RestContext:
    """Holds the incoming request and a queue that receives response chunks."""

    request: ChatRequest
    agent_version: str
    _queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue, init=False)

    def close(self) -> None:
        """Signal end-of-response to any consumer of this context."""
        self._queue.put_nowait(None)

    async def collect(self) -> str:
        """Drain the queue into a single string (used by /invoke)."""
        parts: list[str] = []
        async for chunk in self:
            parts.append(chunk)
        return "".join(parts)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[str]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk


class RestApiInterface(BaseInterface[_RestContext]):
    """REST API frontend — runs every request through the full BaseInterface pipeline.

    Both /chat/invoke and /chat/stream benefit from reflexes, structured logging,
    run lifecycle management, and error handling defined in BaseInterface._execute_run.
    """

    def __init__(
        self, checkpointer: BaseCheckpointSaver, storage: FileStorage | None = None
    ) -> None:
        super().__init__(checkpointer)
        self._storage = storage

    async def receive_message(self, context: _RestContext) -> IncomingMessage | None:
        req = context.request
        parts: list[ContentPart] = await self._attachment_parts(req)
        if req.message:
            parts.append(TextContent(text=req.message))
        return IncomingMessage(
            parts=parts,
            interface="rest_api",
            sender_id=req.thread_id,
            thread_id=req.thread_id,
            agent_version=context.agent_version,
            # Without an explicit user, the thread is the caller.
            user_id=req.user_id or req.thread_id,
        )

    async def _attachment_parts(self, req: ChatRequest) -> list[ContentPart]:
        """Copy uploaded files (POST /files/upload) next to Telegram's uploads, as file
        parts — the agent sees them exactly as it sees a file sent on Telegram."""
        parts: list[ContentPart] = []
        for attachment in req.attachments:
            data = await self._storage.read(attachment.file_id) if self._storage else None
            if data is None:
                parts.append(
                    TextContent(text=f"[Attachment {attachment.file_id} could not be found]")
                )
                continue
            meta = await self._storage.get_metadata(attachment.file_id)
            filename = Path(meta.filename).name if meta else attachment.file_id
            mime = attachment.mime_type or mimetypes.guess_type(filename)[0]
            part = FileContent(
                path=str(
                    UPLOADS_DIR / f"rest-{req.thread_id}" / f"{attachment.file_id[:8]}_{filename}"
                ),
                mime_type=mime or "application/octet-stream",
            )
            await part.write(data)
            parts.append(part)
        return parts

    async def send_stream(self, stream: AsyncIterator[AgentEvent], context: _RestContext) -> None:
        async for event in stream:
            if isinstance(event, ChatModelStreamEvent) and event.delta:
                context._queue.put_nowait(event.delta)

    async def send_message(self, message: str, context: _RestContext) -> None:
        context._queue.put_nowait(message)

    async def send_notification(self, target_id: str, text: str) -> None:
        pass  # REST has no push channel — reminders set during REST sessions are silently dropped

    async def resolve_thread(
        self,
        thread_id: str,
        *,
        topic_name: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """Return *thread_id* unchanged; raise for logical aliases unsupported by REST.

        REST has no persistent connection, so ``"default"`` and ``"new"`` have no
        meaning.  Only concrete thread IDs are valid here.
        """
        if thread_id in ("default", "new"):
            raise ValueError(
                f"REST API does not support {thread_id!r} thread resolution — "
                "push delivery requires a Telegram interface."
            )
        return thread_id

    def conversation_id(self, thread_id: str) -> str:
        """REST thread IDs are stable and client-chosen, so each is its own conversation."""
        return f"rest-{thread_id}"

    async def send_proactive(self, thread_id: str, text: str) -> None:
        """No-op: REST has no push channel for forward-type pushes."""

    async def announcement_threads(self) -> list[str]:
        """No targets: REST has no push channel, so announcements have nowhere to land."""
        return []

    async def send_proactive_stream(
        self, thread_id: str, stream: AsyncIterator[AgentEvent]
    ) -> None:
        """Consume and discard the stream; REST has no push channel."""
        async for _ in stream:
            pass

    async def request_approval(self, request: ApprovalRequest, context: _RestContext) -> None:
        context._queue.put_nowait(
            f"\n⏸ Approval required: [{request.tool_name}] `{request.description}`. "
            f"Use POST /chat/{{thread_id}}/approve with "
            f'{{"decision": "approved_once" | "approved_always" | "denied"}}.'
        )

    async def invoke(self, request: ChatRequest, agent_version: str) -> str:
        """Run the full pipeline and return the complete response text."""
        ctx = _RestContext(request=request, agent_version=agent_version)
        task = asyncio.create_task(self._run_and_close(ctx))
        text = await ctx.collect()
        await task
        return text

    async def stream_sse(self, request: ChatRequest, agent_version: str) -> AsyncIterator[str]:
        """Run the full pipeline, yielding Server-Sent Events."""
        ctx = _RestContext(request=request, agent_version=agent_version)
        task = asyncio.create_task(self._run_and_close(ctx))
        try:
            async for chunk in ctx:
                yield f"event: text_delta\ndata: {json.dumps({'delta': chunk})}\n\n"
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        done_payload = json.dumps({"thread_id": request.thread_id, "agent": agent_version})
        yield f"event: done\ndata: {done_payload}\n\n"

    async def invoke_resume(
        self,
        thread_id: str,
        agent_version: str,
        sender_id: str,
        decision: ApprovalDecision,
    ) -> str:
        """Resume a paused approval and return the complete response text."""
        ctx = _RestContext(
            request=ChatRequest(thread_id=thread_id, message="", agent=agent_version),
            agent_version=agent_version,
        )
        task = asyncio.create_task(
            self._resume_and_close(ctx, thread_id, agent_version, sender_id, decision)
        )
        text = await ctx.collect()
        await task
        return text

    async def _run_and_close(self, ctx: _RestContext) -> None:
        try:
            await self.run(ctx)
        finally:
            ctx.close()

    async def _resume_and_close(
        self,
        ctx: _RestContext,
        thread_id: str,
        agent_version: str,
        sender_id: str,
        decision: ApprovalDecision,
    ) -> None:
        try:
            await self._execute_resume(
                thread_id=thread_id,
                agent_version=agent_version,
                sender_id=sender_id,
                interface="rest_api",
                decision=decision,
                context=ctx,
            )
        finally:
            ctx.close()
