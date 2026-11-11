"""Fake LLM + fake Telegram Bot API for the local scenario stack. Dev only.

One process, two surfaces:

  /v1/...                 an OpenAI-compatible chat-completions server.  Each reply
                          comes from a script the scenario runner uploads, so every
                          model call is free and deterministic.
  /bot<token>/<method>    a Telegram Bot API server.  AUG's bot points here through
                          TELEGRAM_BASE_URL; updates are queued by the runner and every
                          outgoing call is recorded.
  /control/...            what scripts/scenarios.py uses to drive and read both.

How a reply is picked: the runner uploads ``scripts = {"s1": [step, step, ...]}``.  A
request that offers tools is matched to the last user message containing ``#<key>``;
the step is the number of assistant messages after that user message.  Requests
without tools (compaction, consolidation) get a fixed summary.

Step shapes:
  {"text": "...", "chunk_delay": 0.0}
  {"tool": "name", "args": {...}}              (optionally with "text")
  {"error": 429}                               rate limit
  {"error": 400, "code": "context_length_exceeded"}
  {"error": 401}                               bad credentials

Run: ``python scripts/fakes.py`` (port 8081).
"""

import asyncio
import json
import re
import time
from itertools import count
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

_BOT_USER = {"id": 999000, "is_bot": True, "first_name": "AUG", "username": "aug_fake_bot"}
_SUMMARY = "Summary of the conversation so far (fake)."


class _State:
    def __init__(self) -> None:
        self.scripts: dict[str, list[dict[str, Any]]] = {}
        self.llm_requests: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.update_ids = count(1)
        self.message_ids = count(5000)
        self.call_ids = count(1)
        self.new_update = asyncio.Event()
        self.polled = False

    def reset(self, scripts: dict[str, list[dict[str, Any]]]) -> None:
        # Keep the event and the poll flag: a long poll already waiting holds this event.
        event, polled = self.new_update, self.polled
        self.__init__()
        self.scripts, self.new_update, self.polled = scripts, event, polled


state = _State()


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


@app.post("/control/reset")
async def control_reset(request: Request) -> dict:
    body = await request.json()
    state.reset(body.get("scripts", {}))
    return {"ok": True}


@app.post("/control/scripts")
async def control_scripts(request: Request) -> dict:
    state.scripts.update((await request.json())["scripts"])
    return {"ok": True}


@app.post("/control/telegram/update")
async def control_update(request: Request) -> dict:
    update = await request.json()
    update["update_id"] = next(state.update_ids)
    state.updates.append(update)
    state.new_update.set()
    return {"update_id": update["update_id"]}


@app.get("/control/telegram/calls")
async def control_calls() -> list[dict]:
    return state.calls


@app.get("/control/telegram/polling")
async def control_polling() -> dict:
    return {"polling": state.polled}


@app.get("/control/llm/requests")
async def control_llm_requests() -> list[dict]:
    return state.llm_requests


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    key, step_index = _match_script(messages) if body.get("tools") else (None, 0)
    step: dict[str, Any] = (
        state.scripts[key][step_index]
        if key is not None and step_index < len(state.scripts[key])
        else {"text": _SUMMARY if not body.get("tools") else "(script exhausted)"}
    )
    state.llm_requests.append(
        {
            "script": key,
            "step": step_index,
            "tools": bool(body.get("tools")),
            "roles": [m.get("role") for m in messages],
        }
    )
    if "error" in step:
        return _error_response(step)
    if body.get("stream"):
        return StreamingResponse(
            _stream(step, body.get("model", "")), media_type="text/event-stream"
        )
    return JSONResponse(_completion(step, body.get("model", "")))


@app.post("/v1/audio/transcriptions")
async def transcriptions() -> dict:
    return {"text": "fake transcript"}


def _match_script(messages: list[dict]) -> tuple[str | None, int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        text = _text_of(messages[i].get("content"))
        for key in state.scripts:
            if re.search(rf"#{re.escape(key)}\b", text):
                after = messages[i + 1 :]
                return key, sum(1 for m in after if m.get("role") == "assistant")
    return None, 0


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _tool_calls(step: dict) -> list[dict]:
    calls = step.get("tools") or (
        [{"name": step["tool"], "args": step.get("args", {})}] if "tool" in step else []
    )
    return [
        {
            "index": i,
            "id": f"call_{next(state.call_ids)}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))},
        }
        for i, c in enumerate(calls)
    ]


def _completion(step: dict, model: str) -> dict:
    tool_calls = _tool_calls(step)
    message: dict[str, Any] = {"role": "assistant", "content": step.get("text", "")}
    if tool_calls:
        message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in tool_calls]
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def _stream(step: dict, model: str):
    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    text = step.get("text", "")
    delay = step.get("chunk_delay", 0.0)
    words = text.split(" ")
    for i, word in enumerate(words):
        yield chunk({"content": word if i == len(words) - 1 else word + " "})
        if delay:
            await asyncio.sleep(delay)
    tool_calls = _tool_calls(step)
    if tool_calls:
        yield chunk({"tool_calls": tool_calls})
    yield chunk({}, "tool_calls" if tool_calls else "stop")
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    final = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": model}
    final.update(created=int(time.time()), choices=[], usage=usage)
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _error_response(step: dict) -> JSONResponse:
    status = step["error"]
    code = step.get("code")
    kinds = {429: "rate_limit_error", 401: "authentication_error", 400: "invalid_request_error"}
    message = {
        429: "Rate limit exceeded (fake).",
        401: "Invalid API key (fake).",
        400: "This model's maximum context length is 200000 tokens (fake).",
    }.get(status, "Fake error.")
    return JSONResponse(
        {"error": {"message": message, "type": kinds.get(status, "api_error"), "code": code}},
        status_code=status,
        headers={"retry-after-ms": "10"},
    )


# ---------------------------------------------------------------------------
# Fake Telegram Bot API
# ---------------------------------------------------------------------------


@app.post("/bot{token}/{method}")
async def bot_method(token: str, method: str, request: Request) -> dict:
    params = await _params(request)
    if method == "getUpdates":
        return {"ok": True, "result": await _get_updates(params)}
    if method not in ("getMe",):
        state.calls.append({"method": method, "params": params, "at": time.time()})
    return {"ok": True, "result": _result(method, params)}


async def _params(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        raw = await request.json()
    else:
        form = await request.form()
        raw = {
            k: v if isinstance(v, str) else f"<file {getattr(v, 'filename', '')}>"
            for k, v in form.items()
        }
    params: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str) and v[:1] in "{[":
            try:
                v = json.loads(v)
            except ValueError:
                pass
        params[k] = v
    return params


async def _get_updates(params: dict) -> list[dict]:
    offset = int(params.get("offset") or 0)
    timeout = float(params.get("timeout") or 0)
    deadline = time.time() + min(timeout, 10)
    state.polled = True
    while True:
        state.new_update.clear()  # before the check: an update added after it sets it again
        pending = [u for u in state.updates if u["update_id"] >= offset]
        if pending or time.time() >= deadline:
            return pending
        try:
            await asyncio.wait_for(state.new_update.wait(), max(0.0, deadline - time.time()))
        except TimeoutError:
            pass


def _result(method: str, params: dict) -> Any:
    if method == "getMe":
        return _BOT_USER
    if method in (
        "sendMessage",
        "sendPhoto",
        "sendDocument",
        "sendAudio",
        "sendVideo",
        "sendVoice",
    ):
        message = _message(params, next(state.message_ids))
        params["_message_id"] = message["message_id"]
        return message
    if method in ("editMessageText", "editMessageReplyMarkup"):
        return _message(params, int(params.get("message_id") or 0))
    if method == "createForumTopic":
        return {
            "message_thread_id": next(state.message_ids),
            "name": params.get("name", ""),
            "icon_color": 0,
        }
    if method == "getFile":
        return {"file_id": params.get("file_id"), "file_unique_id": "u", "file_path": "file.bin"}
    return True


def _message(params: dict, message_id: int) -> dict:
    chat_id = int(params.get("chat_id") or 0)
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": "private" if chat_id > 0 else "supergroup"},
        "from": _BOT_USER,
    }
    if "text" in params:
        message["text"] = params["text"]
    if params.get("message_thread_id"):
        message["message_thread_id"] = int(params["message_thread_id"])
    if params.get("reply_markup"):
        message["reply_markup"] = params["reply_markup"]
    return message


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="warning")
