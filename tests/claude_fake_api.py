"""A fake Anthropic Messages API for running the real Claude Code CLI in tests, free.

The CLI accepts ``ANTHROPIC_BASE_URL``; this server answers every ``/v1/messages`` call
with whatever ``respond(request)`` returns: a list of content blocks (text and tool_use),
streamed back the way the API streams them.  Every request is kept in ``requests``.
"""

import json
import socketserver
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Respond = Callable[[dict[str, Any]], list[dict[str, Any]]]


class FakeAnthropicAPI:
    def __init__(self, respond: Respond) -> None:
        self.respond = respond
        self.requests: list[dict[str, Any]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                api.requests.append(body)
                blocks = api.respond(body)
                data = _sse(body.get("model", ""), blocks)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "FakeAnthropicAPI":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()


class _Server(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer's own bind looks up the host's FQDN, which can take half a minute.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def last_user_content(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The content blocks of the request's last user message."""
    for message in reversed(request.get("messages", [])):
        if message["role"] == "user":
            content = message["content"]
            return [{"type": "text", "text": content}] if isinstance(content, str) else content
    return []


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(tool_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": args}


def _sse(model: str, blocks: list[dict[str, Any]]) -> bytes:
    message = {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            events += [
                {"type": "content_block_start", "index": index, "content_block": text("")},
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            ]
        else:
            start = {**block, "input": {}}
            events += [
                {"type": "content_block_start", "index": index, "content_block": start},
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            ]
        events.append({"type": "content_block_stop", "index": index})
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
