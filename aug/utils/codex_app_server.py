"""Async client for ``codex app-server``: JSON-RPC over the process's stdin and stdout.

How to talk to the Codex app-server, nothing about what to say: start the process, send
requests and await their results, receive its notifications, and answer the requests it
sends the client (a tool call, say).  Requests from the server are answered by their own
task, so one that takes half an hour holds nothing else up.
"""

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import count
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_server_logger = logging.getLogger("aug.codex_app_server")

# One JSON-RPC message per line; a tool result with an image can be megabytes.
_LINE_LIMIT = 64 * 1024 * 1024
_CLOSE_SECONDS = 5
_START_SECONDS = 60
# Every request is answered at once (a turn's work arrives as notifications), so one
# unanswered this long means the server is stuck.
_REQUEST_SECONDS = 120
# Lines of the server's stderr kept to explain an exit.
_STDERR_TAIL = 40
# JSON-RPC's own code for a method the receiver does not have.
METHOD_NOT_FOUND = -32601

JsonObject = dict[str, Any]
# A notification: its method and params.  Called on the reader; must not block.
NotificationHandler = Callable[[str, JsonObject], None]
# A request from the server: its method and params; returns the result, or raises
# CodexRpcError to answer with an error.
RequestHandler = Callable[[str, JsonObject], Awaitable[JsonObject]]


class CodexRpcError(Exception):
    """An error response, from the server or for it."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class AppServerClosed(Exception):
    """The app-server process is gone; a request in flight will never be answered."""


class CodexAppServer:
    """One ``codex app-server`` process.

    ``start()`` runs the initialize handshake; ``close()`` ends the process.  Once the
    process exits by itself, every request in flight fails with AppServerClosed and
    ``on_exit`` is called.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        on_notification: NotificationHandler,
        on_request: RequestHandler,
        on_exit: Callable[[], None] | None = None,
        client_name: str = "aug",
        client_version: str = "0",
    ) -> None:
        self._argv = list(argv)
        self._env = dict(env)
        self._cwd = cwd
        self._on_notification = on_notification
        self._on_request = on_request
        self._on_exit = on_exit
        self._client = {"name": client_name, "title": "AUG", "version": client_version}
        self._proc: asyncio.subprocess.Process | None = None
        self._ids = count(1)
        self._pending: dict[int, asyncio.Future[JsonObject]] = {}
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL)
        self._gone = False  # the process has exited or is being closed

    @property
    def alive(self) -> bool:
        return self._reader is not None and not self._gone

    async def start(self) -> JsonObject:
        """Start the process and initialize it.  Returns the initialize result."""
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
            limit=_LINE_LIMIT,
        )
        self._reader = asyncio.create_task(self._read(), name="codex-app-server")
        self._stderr = asyncio.create_task(self._drain_stderr(), name="codex-app-server-err")
        try:
            async with asyncio.timeout(_START_SECONDS):
                result = await self.request(
                    "initialize",
                    {"clientInfo": self._client, "capabilities": {"experimentalApi": True}},
                )
            await self._write({"method": "initialized"})
        except BaseException:
            await self.close()
            raise
        logger.info("codex app-server started pid=%s", self._proc.pid)
        return result

    async def request(self, method: str, params: JsonObject | None = None) -> JsonObject:
        """Send *method* and await its result.  Raises CodexRpcError for an error
        response, AppServerClosed when the process is gone, TimeoutError when it does not
        answer within _REQUEST_SECONDS."""
        if not self.alive:
            raise AppServerClosed(f"{method}: the app-server is not running")
        request_id = next(self._ids)
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with asyncio.timeout(_REQUEST_SECONDS):
                await self._write({"id": request_id, "method": method, "params": params or {}})
                return await future
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        self._gone = True
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), _CLOSE_SECONDS)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        for task in (self._reader, self._stderr, *self._tasks):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(t for t in (self._reader, self._stderr, *self._tasks) if t is not None),
            return_exceptions=True,
        )
        self._fail_pending()

    async def _write(self, message: JsonObject) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise AppServerClosed("the app-server is not running")
        data = (json.dumps(message) + "\n").encode()
        async with self._write_lock:
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise AppServerClosed(str(e)) from e

    async def _read(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while line := await proc.stdout.readline():
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.warning("codex app-server sent a line that is not JSON: %.200r", line)
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("codex app-server reader failed")
        finally:
            self._gone = True
            self._fail_pending()
            if self._proc is proc:  # exited by itself, not closed
                for task in self._tasks:  # AUG tools still running for it
                    task.cancel()
                if self._on_exit is not None:
                    self._on_exit()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), _CLOSE_SECONDS)
                logger.warning(
                    "codex app-server exited code=%s; its last words:\n%s",
                    proc.returncode,
                    "\n".join(self._stderr_tail),
                )

    def _dispatch(self, message: JsonObject) -> None:
        method = message.get("method")
        if isinstance(method, str) and "id" in message:
            task = asyncio.create_task(self._answer(message["id"], method, message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif isinstance(method, str):
            try:
                self._on_notification(method, message.get("params") or {})
            except Exception:
                logger.exception("codex notification handler failed method=%s", method)
        else:
            future = self._pending.get(message.get("id"))  # type: ignore[arg-type]
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"] or {}
                future.set_exception(
                    CodexRpcError(int(error.get("code", 0)), str(error.get("message", "")))
                )
            else:
                future.set_result(message.get("result") or {})

    async def _answer(self, request_id: Any, method: str, message: JsonObject) -> None:
        reply: JsonObject
        try:
            result = await self._on_request(method, message.get("params") or {})
            reply = {"id": request_id, "result": result}
        except CodexRpcError as e:
            reply = {"id": request_id, "error": {"code": e.code, "message": e.message}}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("codex request handler failed method=%s", method)
            reply = {"id": request_id, "error": {"code": -32603, "message": str(e)}}
        with contextlib.suppress(AppServerClosed):
            await self._write(reply)

    async def _drain_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        while line := await proc.stderr.readline():
            text = line.decode(errors="replace").rstrip()
            self._stderr_tail.append(text)
            _server_logger.debug("%s", text)

    def _fail_pending(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AppServerClosed("the app-server exited"))
