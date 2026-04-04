"""REST API interface — adapts HTTP chat requests into the BaseInterface pipeline."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver

from aug.api.interfaces.base import BaseInterface, IncomingMessage, TextContent
from aug.api.schemas.chat import ChatRequest
from aug.core.events import AgentEvent, ChatModelStreamEvent
from aug.core.tools.approval import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)


@dataclass
class _RestContext:
    """Holds the incoming request and a queue that receives response chunks."""

    request: ChatRequest
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

    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        super().__init__(checkpointer)

    async def receive_message(self, context: _RestContext) -> IncomingMessage | None:
        req = context.request
        return IncomingMessage(
            parts=[TextContent(text=req.message)],
            interface="rest_api",
            sender_id=req.thread_id,
            thread_id=req.thread_id,
            agent_version=req.agent,
        )

    async def send_stream(self, stream: AsyncIterator[AgentEvent], context: _RestContext) -> None:
        async for event in stream:
            if isinstance(event, ChatModelStreamEvent) and event.delta:
                context._queue.put_nowait(event.delta)

    async def send_message(self, message: str, context: _RestContext) -> None:
        context._queue.put_nowait(message)

    async def send_notification(self, target_id: str, text: str) -> None:
        pass  # REST has no push channel — reminders set during REST sessions are silently dropped

    async def request_approval(self, request: ApprovalRequest, context: _RestContext) -> None:
        context._queue.put_nowait(
            f"\n⏸ Approval required to run `{request.command}` on `{request.target}`. "
            f"Use POST /chat/approve/{{thread_id}} with "
            f'{{"decision": "approved_once" | "approved_always" | "denied"}}.'
        )

    async def invoke(self, request: ChatRequest) -> str:
        """Run the full pipeline and return the complete response text."""
        ctx = _RestContext(request=request)
        task = asyncio.create_task(self._run_and_close(ctx))
        text = await ctx.collect()
        await task
        return text

    async def stream_sse(self, request: ChatRequest) -> AsyncIterator[str]:
        """Run the full pipeline, yielding Server-Sent Events."""
        ctx = _RestContext(request=request)
        task = asyncio.create_task(self._run_and_close(ctx))
        try:
            async for chunk in ctx:
                yield f"event: text_delta\ndata: {json.dumps({'delta': chunk})}\n\n"
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        done_payload = json.dumps({"thread_id": request.thread_id, "agent": request.agent})
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
            request=ChatRequest(thread_id=thread_id, message="", agent=agent_version)
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
