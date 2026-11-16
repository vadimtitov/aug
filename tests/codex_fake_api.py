"""A fake OpenAI Responses API for running the real Codex app-server in tests, free.

Codex is pointed at it as a custom model provider (``codex_provider_config``).  Every
``/v1/responses`` call is answered with whatever ``respond(request)`` returns: a list of
output items (messages and function calls), streamed back the way the API streams them,
or an ``ApiError``.  Every request is kept in ``requests``.
"""

import json
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from typing import Any

# A model Codex has no metadata for: plain function tools, no code mode.
FAKE_MODEL = "gpt-test"


@dataclass(frozen=True)
class ApiError:
    status: int
    message: str


Respond = Callable[[dict[str, Any]], "list[dict[str, Any]] | ApiError"]


class FakeResponsesAPI:
    def __init__(self, respond: Respond) -> None:
        self.respond = respond
        self.requests: list[dict[str, Any]] = []
        ids = count(1)
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(404)
                self.send_header("content-length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                api.requests.append(body)
                output = api.respond(body)
                if isinstance(output, ApiError):
                    data = json.dumps({"error": {"message": output.message}}).encode()
                    status, content_type = output.status, "application/json"
                else:
                    data = _sse(f"resp_{next(ids)}", output)
                    status, content_type = 200, "text/event-stream"
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "FakeResponsesAPI":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()


def codex_provider_config(url: str) -> dict[str, str]:
    """CodexRuntime settings that send every model call to the fake API at *url*."""
    return {
        "model_provider": '"fake"',
        "model_providers.fake": (
            f'{{name="fake",base_url="{url}",wire_api="responses",env_key="FAKE_API_KEY"}}'
        ),
    }


def message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call(call_id: str, name: str, args: dict[str, Any], namespace: str | None = None) -> dict:
    item = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args),
        "status": "completed",
    }
    if namespace:
        item["namespace"] = namespace
    return item


def last_input(request: dict[str, Any]) -> dict[str, Any]:
    items = request.get("input") or []
    return items[-1] if items else {}


def said(request: dict[str, Any]) -> str:
    """All the user's text in the request."""
    return " ".join(
        c.get("text", "")
        for item in request.get("input") or []
        if item.get("role") == "user"
        for c in item.get("content") or []
    )


def developer_text(request: dict[str, Any]) -> str:
    return " ".join(
        c.get("text", "")
        for item in request.get("input") or []
        if item.get("role") == "developer"
        for c in item.get("content") or []
    )


class _Server(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer's own bind looks up the host's FQDN, which can take half a minute.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def _sse(response_id: str, output: list[dict[str, Any]]) -> bytes:
    events: list[dict[str, Any]] = [{"type": "response.created", "response": {"id": response_id}}]
    for index, item in enumerate(output):
        item = {**item, "id": f"{response_id}_{index}"}
        if item["type"] == "message":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "content": [], "status": "in_progress"},
                }
            )
            for part in item["content"]:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "output_index": index,
                        "content_index": 0,
                        "item_id": item["id"],
                        "delta": part["text"],
                    }
                )
        else:
            events.append(
                {"type": "response.output_item.added", "output_index": index, "item": item}
            )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 200},
        "output_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1010,
    }
    events.append({"type": "response.completed", "response": {"id": response_id, "usage": usage}})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
