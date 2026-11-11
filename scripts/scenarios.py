"""Scenario suite: drive AUG through fake Telegram and REST, write a normalised transcript.

Runs against the scenario stack (docker-compose.scenarios.yml, see ``make scenarios``):
AUG with the dev-only "scripted" agent, whose model and Telegram are both fakes from
scripts/fakes.py.  No real model is called.

The transcript is what a user would see (final Telegram messages, buttons, REST SSE
frames) plus a summary of each model request, with volatile values (ids, timestamps)
removed.  Run it on main and on a branch and diff the two files: "no visible behaviour
change" becomes a diff, not a belief.

    python scripts/scenarios.py seed .scenarios/data
    python scripts/scenarios.py run --out .scenarios/transcript.txt [--only s1,s2]
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

AUG = "http://localhost:8013"
FAKES = "http://localhost:8091"
API_KEY = "scenarios"
USER_ID = 111  # also the DM chat id; matches TELEGRAM_ALLOWED_CHAT_IDS in the compose file
QUIET_SECONDS = 2.5

SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "s1": [{"text": "Hello from the scripted agent."}],
    "s1r": [{"text": "Hello over REST."}],
    "s2": [{"tool": "dev_progress", "args": {"steps": 3}}, {"text": "Progress done."}],
    "s3a": [{"tool": "dev_guarded", "args": {"command": "ls -la"}}, {"text": "Approved path."}],
    "s3b": [{"tool": "dev_guarded", "args": {"command": "rm x"}}, {"text": "Denied path."}],
    "s3c": [{"tool": "dev_guarded", "args": {"command": "push cmd"}}, {"text": "Push path."}],
    "s3r": [{"tool": "dev_guarded", "args": {"command": "rest cmd"}}, {"text": "REST approved."}],
    "s4": [{"tool": "dev_wait", "args": {"seconds": 2}}, {"text": "Saw the extra message."}],
    "s4b": [
        {"text": " ".join(f"word{i}" for i in range(30)), "chunk_delay": 0.1},
        {"text": "Follow-up for the late message."},
    ],
    "s5": [{"tool": "dev_wait", "args": {"seconds": 30}}, {"text": "Should not appear."}],
    "s5b": [{"text": "Still works after stop."}],
    "s6": [{"text": "Fresh context."}],
    "s7a": [{"error": 429}],
    "s7b": [{"error": 400, "code": "context_length_exceeded"}],
    "s7c": [{"error": 401}],
    "s8": [{"text": "Scheduled reminder text."}],
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def s1_reply(d: "Driver") -> None:
    await d.tg_text("hello #s1")
    await d.quiet()
    await d.rest_stream("hi #s1r", context="rest-s1")


async def s2_tool(d: "Driver") -> None:
    await d.tg_text("run progress #s2")
    await d.quiet()


async def s3_approvals(d: "Driver") -> None:
    await d.tg_text("guarded once #s3a")
    await d.quiet()
    await d.tg_press("Run Once")
    await d.quiet()
    await d.tg_text("guarded deny #s3b")
    await d.quiet()
    await d.tg_press("Deny")
    await d.quiet()
    await d.push("agent", "unattended guarded #s3c")
    await d.quiet()
    await d.rest_stream("guarded over rest #s3r", context="rest-s3")
    await d.rest_approve("rest-s3", "approved_once")


async def s4_inject(d: "Driver") -> None:
    await d.tg_text("wait a bit #s4")
    await d.wait_llm(lambda r: r["script"] == "s4")
    await asyncio.sleep(0.5)
    await d.tg_text("extra message mid-run")
    await d.quiet()
    await d.tg_text("stream slowly #s4b")
    await d.wait_llm(lambda r: r["script"] == "s4b")
    await asyncio.sleep(1.0)
    await d.tg_text("late message")
    await d.quiet()


async def s5_stop(d: "Driver") -> None:
    await d.tg_text("long wait #s5")
    await d.wait_llm(lambda r: r["script"] == "s5")
    await asyncio.sleep(1.0)
    await d.tg_text("/stop")
    await d.quiet()
    await d.tg_text("after stop #s5b")
    await d.quiet()


async def s6_compact_clear(d: "Driver") -> None:
    await d.tg_text("/compact")
    await d.quiet()
    await d.tg_text("/clear")
    await d.quiet()
    await d.tg_text("new start #s6")
    await d.quiet()


async def s7_errors(d: "Driver") -> None:
    for key in ("s7a", "s7b", "s7c"):
        await d.tg_text(f"trigger #{key}")
        await d.quiet()


async def s8_push(d: "Driver") -> None:
    await d.push("agent", "scheduled #s8")
    await d.quiet()
    await d.push("forward", "Forwarded notice.")
    await d.quiet()


SCENARIOS: dict[str, Callable[["Driver"], Awaitable[None]]] = {
    "s1": s1_reply,
    "s2": s2_tool,
    "s3": s3_approvals,
    "s4": s4_inject,
    "s5": s5_stop,
    "s6": s6_compact_clear,
    "s7": s7_errors,
    "s8": s8_push,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class Driver:
    """Talks to AUG and the fakes, and turns what came back into transcript lines."""

    def __init__(self, client: httpx.AsyncClient, id_field: str) -> None:
        self._client = client
        self._id_field = id_field
        self._message_ids = iter(range(1, 10**6))
        self.lines: list[str] = []
        self._calls_seen = 0
        self._llm_seen = 0

    async def tg_text(self, text: str) -> None:
        self.lines.append(f"> tg: {text}")
        message: dict[str, Any] = {
            "message_id": next(self._message_ids),
            "date": int(time.time()),
            "chat": {"id": USER_ID, "type": "private"},
            "from": {"id": USER_ID, "is_bot": False, "first_name": "Tester"},
            "text": text,
        }
        if text.startswith("/"):
            message["entities"] = [
                {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
            ]
        await self._fake("POST", "/control/telegram/update", json={"message": message})

    async def tg_press(self, label: str) -> None:
        """Press the most recent inline button whose label contains *label*."""
        self.lines.append(f"> tg press: {label}")
        for call in reversed(await self._calls()):
            markup = call["params"].get("reply_markup") or {}
            for row in markup.get("inline_keyboard", []):
                for button in row:
                    if label in button.get("text", ""):
                        message = {
                            "message_id": call["params"].get("_message_id", 1),
                            "date": int(time.time()),
                            "chat": {"id": USER_ID, "type": "private"},
                            "from": {"id": 999000, "is_bot": True, "first_name": "AUG"},
                            "text": call["params"].get("text", ""),
                        }
                        update = {
                            "callback_query": {
                                "id": str(next(self._message_ids)),
                                "from": {"id": USER_ID, "is_bot": False, "first_name": "Tester"},
                                "chat_instance": "ci",
                                "data": button["callback_data"],
                                "message": message,
                            }
                        }
                        await self._fake("POST", "/control/telegram/update", json=update)
                        return
        self.lines.append(f"! no button labelled {label!r}")

    async def push(self, push_type: str, message: str) -> None:
        self.lines.append(f"> push {push_type}: {message}")
        response = await self._client.post(
            f"{AUG}/hooks/push",
            headers={"X-API-Key": API_KEY},
            json={"interface": "telegram", "message": message, "type": push_type},
        )
        self.lines.append(f"< push status {response.status_code}")

    async def rest_stream(self, message: str, context: str) -> None:
        self.lines.append(f"> rest: {message}")
        body = {"message": message, self._id_field: context, "agent": "scripted"}
        async with self._client.stream(
            "POST", f"{AUG}/chat/stream", headers={"X-API-Key": API_KEY}, json=body, timeout=60
        ) as response:
            if response.status_code != 200:
                await response.aread()
                self.lines.append(f"< rest {response.status_code}: {_normalise(response.text)}")
            event = ""
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: "):
                    self.lines.append(f"< sse {event}: {_normalise(line[len('data: ') :])}")
        await self._llm_lines()

    async def rest_approve(self, context: str, decision: str) -> None:
        self.lines.append(f"> rest approve: {decision}")
        response = await self._client.post(
            f"{AUG}/chat/{context}/approve",
            headers={"X-API-Key": API_KEY},
            json={"decision": decision, "agent": "scripted"},
            timeout=60,
        )
        self.lines.append(f"< approve {response.status_code}: {_normalise(response.text)}")
        await self._llm_lines()

    async def wait_llm(self, predicate: Callable[[dict], bool], limit: float = 30) -> None:
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if any(predicate(r) for r in (await self._llm())[self._llm_seen :]):
                return
            await asyncio.sleep(0.1)
        self.lines.append("! timed out waiting for a model request")

    async def quiet(self, limit: float = 90) -> None:
        """Wait until neither fake has seen a new call for QUIET_SECONDS, then record."""
        deadline = time.monotonic() + limit
        last = (-1, -1)
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            now = (len(await self._calls()), len(await self._llm()))
            if now != last:
                last, last_change = now, time.monotonic()
            elif time.monotonic() - last_change >= QUIET_SECONDS:
                break
            await asyncio.sleep(0.2)
        else:
            self.lines.append("! timed out waiting for quiet")
        await self._tg_lines()
        await self._llm_lines()

    async def _tg_lines(self) -> None:
        calls = await self._calls()
        new, self._calls_seen = calls[self._calls_seen :], len(calls)
        messages: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        for call in new:
            method, params = call["method"], call["params"]
            if method in ("sendMessage", "sendPhoto", "sendDocument"):
                mid = params["_message_id"]
                messages[mid] = {
                    "text": params.get("text") or f"[{method}] {params.get('caption', '')}"
                }
                messages[mid]["markup"] = params.get("reply_markup")
                messages[mid]["silent"] = params.get("disable_notification")
                order.append(mid)
            elif method == "editMessageText":
                mid = int(params.get("message_id") or 0)
                messages.setdefault(mid, {"text": "", "markup": None, "silent": None})
                if mid not in order:
                    order.append(mid)
                messages[mid]["text"] = params.get("text", "")
                messages[mid]["markup"] = params.get("reply_markup")
            elif method == "editMessageReplyMarkup":
                mid = int(params.get("message_id") or 0)
                if mid in messages:
                    messages[mid]["markup"] = params.get("reply_markup")
                else:
                    buttons = _buttons(params.get("reply_markup"))
                    self.lines.append(f"< tg markup edit on earlier message: {buttons}")
            elif method == "deleteMessage":
                mid = int(params.get("message_id") or 0)
                if mid in messages:
                    messages[mid]["text"] = "(deleted) " + messages[mid]["text"]
            elif method == "answerCallbackQuery" and params.get("text"):
                self.lines.append(f"< tg toast: {params['text']}")
        for mid in order:
            m = messages[mid]
            silent = " (silent)" if str(m["silent"]).lower() == "true" else ""
            self.lines.append(f"< tg{silent}: {_normalise(m['text'])}{_buttons(m['markup'])}")

    async def _llm_lines(self) -> None:
        requests = await self._llm()
        new, self._llm_seen = requests[self._llm_seen :], len(requests)
        for r in new:
            kind = f"{r['script']}[{r['step']}]" if r["tools"] else "no-tools"
            self.lines.append(f"  llm {kind}: {','.join(r['roles'])}")

    async def _calls(self) -> list[dict]:
        return (await self._fake("GET", "/control/telegram/calls")).json()

    async def _llm(self) -> list[dict]:
        return (await self._fake("GET", "/control/llm/requests")).json()

    async def _fake(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = await self._client.request(method, f"{FAKES}{path}", **kwargs)
        response.raise_for_status()
        return response


def _buttons(markup: Any) -> str:
    if not markup:
        return ""
    labels = [b.get("text", "") for row in markup.get("inline_keyboard", []) for b in row]
    return f" [{' | '.join(labels)}]" if labels else ""


def _normalise(text: str) -> str:
    text = re.sub(
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?( ?UTC|Z|[+-]\d{2}:\d{2})?", "<time>", text
    )
    text = re.sub(r"call_\d+", "call_<n>", text)
    text = re.sub("[🌑🌘🌗🌖🌕🌔🌓🌒]", "<spin>", text)  # spinner frame depends on timing
    text = re.sub(r"\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<uuid>", text)
    return text.replace("\n", "\\n")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def seed(data_dir: Path) -> None:
    """Fresh data dir: the test DM and the REST threads use the scripted agent."""
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    conversations = {f"tg-{USER_ID}": {"agent": "scripted"}}
    (data_dir / "settings.json").write_text(json.dumps({"conversations": conversations}, indent=2))


async def run(out: Path, only: list[str] | None, id_field: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        await _wait_healthy(client)
        await client.post(f"{FAKES}/control/reset", json={"scripts": SCRIPTS})
        driver = Driver(client, id_field)
        for name, scenario in SCENARIOS.items():
            if only and name not in only:
                continue
            driver.lines.append(f"\n## {name}")
            await scenario(driver)
            print(f"{name} done", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(driver.lines).lstrip() + "\n")
    print(f"transcript: {out}", file=sys.stderr)
    return 0


async def _wait_healthy(client: httpx.AsyncClient, limit: float = 120) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            healthy = (await client.get(f"{AUG}/health")).status_code == 200
            polling = (await client.get(f"{FAKES}/control/telegram/polling")).json()["polling"]
            if healthy and polling:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise SystemExit("AUG scenario stack did not become healthy")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    seed_parser = sub.add_parser("seed")
    seed_parser.add_argument("data_dir", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--only", default="")
    run_parser.add_argument(
        "--id-field", default="context_id", help="REST request field naming the chat"
    )
    args = parser.parse_args()
    if args.command == "seed":
        seed(args.data_dir)
        return 0
    only = [s for s in args.only.split(",") if s] or None
    return asyncio.run(run(args.out, only, args.id_field))


if __name__ == "__main__":
    sys.exit(main())
