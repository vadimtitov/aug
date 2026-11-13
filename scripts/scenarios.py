"""Scenario suite: drive AUG through fake Telegram and REST, write a normalised transcript.

Runs against the scenario stack (docker-compose.scenarios.yml, see ``make scenarios``):
AUG with the dev-only "scripted" agent, whose model and Telegram are both fakes from
scripts/fakes.py.  No real model is called.

The transcript is what a user would see (final Telegram messages, buttons, REST SSE
frames) plus a summary of each model request, with volatile values (ids, timestamps)
removed.  Run it on main and on a branch and diff the two files: "no visible behaviour
change" becomes a diff, not a belief.

    python scripts/scenarios.py seed .scenarios/data [--agent v13]
    python scripts/scenarios.py run --out .scenarios/transcript.txt [--only s1,s2] [--agent v13]

``--agent`` runs the suite on another agent version (the scenario stack gives the real
profiles the scripted agent's dev tools); seed and run must name the same one.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import re
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

AUG = "http://localhost:8013"
FAKES = "http://localhost:8091"
API_KEY = "scenarios"
USER_ID = 111  # also the DM chat id; matches TELEGRAM_ALLOWED_CHAT_IDS in the compose file
QUIET_SECONDS = 2.5
_ANIMATION = ("sendMessageDraft", "sendChatAction")  # repeats while a tool runs
_SENT_FILE = {
    "sendPhoto": "photo",
    "sendDocument": "document",
    "sendAudio": "audio",
    "sendVoice": "voice",
    "sendVideo": "video",
    "sendAnimation": "animation",
}
# Files the scripted agent sends in S10, under the data dir (/app/data in the container).
FIXTURES = "/app/data/fixtures"
# The same data dir as seen from here (the compose file mounts it into the container).
HOST_DATA = Path(".scenarios/data")
# Chats seeded on agent versions that no longer exist; startup moves them (S0).
_LEGACY_CHAT = "rest-legacy"
_LEGACY_GLM_CHAT = "rest-legacy-glm"

SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "s1": [{"text": "Hello from the scripted agent."}],
    "s1r": [{"text": "Hello over REST."}],
    "s2": [{"tool": "dev_progress", "args": {"steps": 3}}, {"text": "Progress done."}],
    "s3a": [{"tool": "dev_guarded", "args": {"command": "ls -la"}}, {"text": "Approved path."}],
    "s3b": [{"tool": "dev_guarded", "args": {"command": "rm x"}}, {"text": "Denied path."}],
    "s3c": [{"tool": "dev_guarded", "args": {"command": "push cmd"}}, {"text": "Push path."}],
    "s3d": [
        {
            "tools": [
                {"name": "dev_guarded", "args": {"command": "par-alpha"}},
                {"name": "dev_guarded", "args": {"command": "par-beta"}},
            ]
        },
        {"text": "Parallel path."},
    ],
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
    "s9a": [{"text": "Nice photo."}],
    "s9b": [{"text": "Converted the HEIC."}],
    "s9c": [{"text": "Three photos."}],
    "s9d": [{"text": "Heard you."}],
    "s9e": [{"text": "Got the PDF."}],
    "s9f": [{"text": "Got the video."}],
    "s9r": [{"text": "Saw the attachment."}],
    "s10": [
        {"tool": "respond_with_file", "args": {"path": f"{FIXTURES}/chart.png", "caption": "Q3"}},
        {"tool": "respond_with_file", "args": {"path": f"{FIXTURES}/report.pdf"}},
        {"tool": "respond_with_file", "args": {"path": f"{FIXTURES}/clip.mp3"}},
        {"tool": "respond_with_file", "args": {"path": "/app/data/settings.json"}},
        {"text": "Sent three files."},
    ],
    "s11a": [
        {"tool": "note", "args": {"content": "The tester likes scripted replies."}},
        {"text": "Noted."},
    ],
    "s11b": [
        {
            "tool": "save_skill",
            "args": {"name": "s11-skill", "description": "Added mid-chat", "body": "Do it."},
        },
        {"text": "Skill saved."},
    ],
    "s11c": [{"text": "Third turn."}],
    "s11d": [{"text": "New session."}],
    "s10r": [
        {"tool": "respond_with_file", "args": {"path": f"{FIXTURES}/report.pdf"}},
        {"text": "Report attached."},
    ],
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
    await d.tg_text("two guarded at once #s3d")
    await d.quiet()
    await d.tg_press("Deny", on="par-alpha")
    await d.quiet()
    await d.tg_press("Run Once", on="par-beta")
    await d.quiet()
    await d.rest_stream("guarded over rest #s3r", context="rest-s3", approve="approved_once")


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


async def s9_files_in(d: "Driver") -> None:
    await d.tg_photo(_image((3000, 2000), "JPEG"), caption="what is this #s9a")
    await d.quiet()
    await d.tg_document(_image((4032, 3024), "HEIF"), "IMG_0042.HEIC", "image/heic", "#s9b")
    await d.quiet()
    colours = ((200, 0, 0), (0, 200, 0), (0, 0, 200))
    album = [_image((800, 600), "JPEG", colour) for colour in colours]
    await d.tg_album(album, caption="our trip #s9c")
    await d.quiet()
    await d.tg_voice(b"OggS fake voice", caption="#s9d")
    await d.quiet()
    await d.tg_document(b"%PDF-1.4 fake", "invoice.pdf", "application/pdf", "#s9e")
    await d.quiet()
    await d.tg_video(b"fake mp4", caption="#s9f")
    await d.quiet()
    file_id = await d.rest_upload(_image((1000, 1000), "PNG"), "diagram.png", "image/png")
    await d.rest_stream("look at this #s9r", context="rest-s9", attachments=[file_id])


async def s10_files_out(d: "Driver") -> None:
    await d.tg_text("send me the files #s10")
    await d.quiet()
    await d.rest_stream("report over rest #s10r", context="rest-s10", download=True)


async def s11_session_prompt(d: "Driver") -> None:
    await d.tg_text("/clear")
    await d.quiet()
    for key in ("s11a", "s11b", "s11c"):
        await d.tg_text(f"turn #{key}")
        await d.quiet()
    session = [
        r["system"] for r in await d.llm_requests() if r["script"] in ("s11a", "s11b", "s11c")
    ]
    d.note(f"model calls in the session: {len(session)}")
    d.note(f"system prompt the same on every call: {_yes(len(set(session)) == 1)}")
    last = session[-1]
    d.note(f"the note stayed out of this session's prompt: {_yes('scripted replies' not in last)}")
    d.note(f"the new skill stayed out of this session's prompt: {_yes('s11-skill' not in last)}")
    await d.tg_text("/prompt")
    await d.quiet()
    shown = await d.last_document_sha()
    d.note(f"/prompt is what the model got: {_yes(shown == _sha(session[-1]))}")
    await d.tg_text("/clear")
    await d.quiet()
    await d.tg_text("after clear #s11d")
    await d.quiet()
    fresh = [r["system"] for r in await d.llm_requests() if r["script"] == "s11d"]
    d.note(
        f"the next session lists the new skill: {_yes('- s11-skill: Added mid-chat' in fresh[0])}"
    )


async def s12_dream(d: "Driver") -> None:
    before = _workspace_hashes()
    await d.tg_text("/consolidate")
    await d.quiet()
    after = _workspace_hashes()
    changed = sorted(p for p in after if before.get(p) != after[p])
    d.note(f"files the dream changed: {', '.join(changed) or '(none)'}")
    inside = bool(changed) and all(p.startswith("memory/") for p in changed)
    d.note(f"all inside memory/: {_yes(inside)}")
    index = (HOST_DATA / "workspace" / "memory" / "MEMORY.md").read_text()
    d.note(f"MEMORY.md: {_normalise(index.strip())}")


async def s0_boot(d: "Driver") -> None:
    conversations = json.loads((HOST_DATA / "settings.json").read_text())["conversations"]
    for chat in (_LEGACY_CHAT, _LEGACY_GLM_CHAT):
        d.note(f"chat left on a deleted version now on: {conversations[chat]['agent']}")


SCENARIOS: dict[str, Callable[["Driver"], Awaitable[None]]] = {
    "s0": s0_boot,
    "s1": s1_reply,
    "s2": s2_tool,
    "s3": s3_approvals,
    "s4": s4_inject,
    "s5": s5_stop,
    "s6": s6_compact_clear,
    "s7": s7_errors,
    "s8": s8_push,
    "s9": s9_files_in,
    "s10": s10_files_out,
    "s11": s11_session_prompt,
    "s12": s12_dream,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class Driver:
    """Talks to AUG and the fakes, and turns what came back into transcript lines."""

    def __init__(self, client: httpx.AsyncClient, id_field: str, agent: str) -> None:
        self._client = client
        self._agent = agent
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

    async def tg_photo(self, data: bytes, caption: str = "", group: str = "") -> None:
        width, height = Image.open(io.BytesIO(data)).size
        file_id = await self._tg_file(data)
        photo = [{"file_id": file_id, "file_unique_id": file_id, "width": width, "height": height}]
        self.lines.append(f"> tg photo {width}x{height}: {caption}")
        await self._tg_media({"photo": photo}, caption, group)

    async def tg_album(self, photos: list[bytes], caption: str) -> None:
        """An album: one update per photo, the caption on the first, as Telegram sends it."""
        self.lines.append(f"> tg album of {len(photos)}")
        for i, data in enumerate(photos):
            await self.tg_photo(data, caption if i == 0 else "", group="album-1")

    async def tg_document(self, data: bytes, filename: str, mime_type: str, caption: str) -> None:
        file_id = await self._tg_file(data)
        document = {
            "file_id": file_id,
            "file_unique_id": file_id,
            "file_name": filename,
            "mime_type": mime_type,
        }
        self.lines.append(f"> tg document {filename} ({mime_type}): {caption}")
        await self._tg_media({"document": document}, caption)

    async def tg_voice(self, data: bytes, caption: str) -> None:
        file_id = await self._tg_file(data)
        voice = {
            "file_id": file_id,
            "file_unique_id": file_id,
            "duration": 2,
            "mime_type": "audio/ogg",
        }
        self.lines.append(f"> tg voice: {caption}")
        await self._tg_media({"voice": voice}, caption)

    async def tg_video(self, data: bytes, caption: str) -> None:
        file_id = await self._tg_file(data)
        video = {
            "file_id": file_id,
            "file_unique_id": file_id,
            "width": 640,
            "height": 360,
            "duration": 3,
            "file_name": "clip.mp4",
            "mime_type": "video/mp4",
        }
        self.lines.append(f"> tg video: {caption}")
        await self._tg_media({"video": video}, caption)

    async def rest_upload(self, data: bytes, filename: str, mime_type: str) -> str:
        response = await self._client.post(
            f"{AUG}/files/upload",
            headers={"X-API-Key": API_KEY},
            files={"file": (filename, data, mime_type)},
        )
        body = response.json()
        self.lines.append(f"> rest upload {filename}: {response.status_code} {body['mime_type']}")
        return body["file_id"]

    async def tg_press(self, label: str, on: str = "") -> None:
        """Press the most recent inline button whose label contains *label*, on a message
        whose text contains *on*."""
        self.lines.append(f"> tg press: {label}" + (f" on {on}" if on else ""))
        for call in reversed(await self._calls()):
            if on not in call["params"].get("text", ""):
                continue
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

    async def rest_stream(
        self,
        message: str,
        context: str,
        approve: str = "",
        attachments: list[str] | None = None,
        download: bool = False,
    ) -> None:
        """Stream a REST turn, answering each approval_requested frame with *approve*,
        and fetching each file frame's url when *download*."""
        self.lines.append(f"> rest: {message}")
        body: dict[str, Any] = {"message": message, self._id_field: context, "agent": self._agent}
        if attachments:
            body["attachments"] = [{"file_id": f} for f in attachments]
        urls: list[str] = []
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
                    data = line[len("data: ") :]
                    self.lines.append(f"< sse {event}: {_normalise(data)}")
                    if event == "approval_requested" and approve:
                        await self._rest_approve(context, json.loads(data)["id"], approve)
                    if event == "file":
                        urls.append(json.loads(data)["url"])
        for url in urls if download else []:
            await self._rest_download(url)
        await self._llm_lines()

    async def _rest_download(self, url: str) -> None:
        anonymous = await self._client.get(f"{AUG}{url}")
        response = await self._client.get(f"{AUG}{url}", headers={"X-API-Key": API_KEY})
        self.lines.append(
            f"< download without key {anonymous.status_code}; with key {response.status_code} "
            f"{response.headers.get('content-type')} {len(response.content)} bytes"
        )

    async def _rest_approve(self, context: str, approval_id: str, decision: str) -> None:
        self.lines.append(f"> rest approve: {decision}")
        response = await self._client.post(
            f"{AUG}/chat/{context}/approve",
            headers={"X-API-Key": API_KEY},
            json={"id": approval_id, "decision": decision},
            timeout=60,
        )
        self.lines.append(f"< approve {response.status_code}")

    async def wait_llm(self, predicate: Callable[[dict], bool], limit: float = 30) -> None:
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if any(predicate(r) for r in (await self._llm())[self._llm_seen :]):
                return
            await asyncio.sleep(0.1)
        self.lines.append("! timed out waiting for a model request")

    async def quiet(self, limit: float = 90) -> None:
        """Wait until neither fake has seen a new call for QUIET_SECONDS, then record.

        Drafts and typing don't count: a tool waiting for approval keeps its spinner
        going until the user answers."""
        deadline = time.monotonic() + limit
        last = (-1, -1)
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            calls = [c for c in await self._calls() if c["method"] not in _ANIMATION]
            now = (len(calls), len(await self._llm()))
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
            if method == "sendMessage" or method in _SENT_FILE:
                mid = params["_message_id"]
                text = params.get("text") or ""
                if method in _SENT_FILE:
                    sent = params.get(_SENT_FILE[method], "")
                    text = f"[{method} {sent}] {params.get('caption', '')}".rstrip()
                messages[mid] = {"text": text}
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
            if r.get("images"):
                self.lines.append(f"    images: {', '.join(r['images'])}")
            for hint in r.get("hints", []):
                self.lines.append(f"    hint: {_normalise(hint)}")

    async def _tg_file(self, data: bytes) -> str:
        file_id = f"f{next(self._message_ids)}"
        payload = {"file_id": file_id, "data": base64.b64encode(data).decode()}
        await self._fake("POST", "/control/telegram/file", json=payload)
        return file_id

    async def _tg_media(self, media: dict[str, Any], caption: str, group: str = "") -> None:
        message: dict[str, Any] = {
            "message_id": next(self._message_ids),
            "date": int(time.time()),
            "chat": {"id": USER_ID, "type": "private"},
            "from": {"id": USER_ID, "is_bot": False, "first_name": "Tester"},
            **media,
        }
        if caption:
            message["caption"] = caption
        if group:
            message["media_group_id"] = group
        await self._fake("POST", "/control/telegram/update", json={"message": message})

    async def _calls(self) -> list[dict]:
        return (await self._fake("GET", "/control/telegram/calls")).json()

    def note(self, line: str) -> None:
        """A line the scenario itself checked, rather than one AUG sent."""
        self.lines.append(f"= {line}")

    async def llm_requests(self) -> list[dict]:
        return await self._llm()

    async def last_document_sha(self) -> str:
        documents = [c for c in await self._calls() if c["method"] == "sendDocument"]
        return documents[-1]["params"].get("_sha256", "") if documents else ""

    async def _llm(self) -> list[dict]:
        return (await self._fake("GET", "/control/llm/requests")).json()

    async def _fake(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = await self._client.request(method, f"{FAKES}{path}", **kwargs)
        response.raise_for_status()
        return response


def _image(size: tuple[int, int], fmt: str, colour: tuple[int, int, int] = (90, 140, 200)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format=fmt)
    return out.getvalue()


def _yes(ok: bool) -> str:
    return "yes" if ok else "NO"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _workspace_hashes() -> dict[str, str]:
    """Every file in the workspace (bar the file store) by path, with its hash."""
    root = HOST_DATA / "workspace"
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] != "files"
    }


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
    text = re.sub(r"\bap:[0-9a-f]{12}:", "ap:<id>:", text)
    text = re.sub(r'"id": "[0-9a-f]{12}"', '"id": "<id>"', text)
    text = re.sub(r'"id": "[0-9a-f]{16}"', '"id": "<file id>"', text)
    text = re.sub(r"/files/[0-9a-f]{16}/", "/files/<file id>/", text)
    # Stored files: /app/data/workspace/files/inbox/2026-11/<id>-name → <inbox>/name
    text = re.sub(
        r"/app/data/workspace/files/(inbox|outbox)/\d{4}-\d{2}/[0-9a-f]{16}-", r"<\1>/", text
    )
    return text.replace("\n", "\\n")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def seed(data_dir: Path, agent: str) -> None:
    """Fresh data dir: the test DM and the REST threads use *agent*; one more chat is
    left on an agent version that no longer exists."""
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    conversations = {
        f"tg-{USER_ID}": {"agent": agent},
        _LEGACY_CHAT: {"agent": "v11_kimi"},
        _LEGACY_GLM_CHAT: {"agent": "v12_glm5"},
    }
    (data_dir / "settings.json").write_text(json.dumps({"conversations": conversations}, indent=2))
    # The dreams count as run, so none starts at boot behind the scenarios' back.
    ran = "2999-01-01T00:00:00+00:00"
    (data_dir / "state.json").write_text(
        json.dumps({"jobs": {"dream_light": ran, "dream_deep": ran}})
    )
    fixtures = data_dir / "fixtures"
    fixtures.mkdir()
    (fixtures / "chart.png").write_bytes(_image((640, 480), "PNG"))
    (fixtures / "report.pdf").write_bytes(b"%PDF-1.4 fake report")
    (fixtures / "clip.mp3").write_bytes(b"ID3 fake audio")


async def run(out: Path, only: list[str] | None, id_field: str, agent: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        await _wait_healthy(client)
        await client.post(f"{FAKES}/control/reset", json={"scripts": SCRIPTS})
        driver = Driver(client, id_field, agent)
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
    seed_parser.add_argument("--agent", default="scripted")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--only", default="")
    run_parser.add_argument(
        "--id-field", default="context_id", help="REST request field naming the chat"
    )
    run_parser.add_argument("--agent", default="scripted")
    args = parser.parse_args()
    if args.command == "seed":
        seed(args.data_dir, args.agent)
        return 0
    only = [s for s in args.only.split(",") if s] or None
    return asyncio.run(run(args.out, only, args.id_field, args.agent))


if __name__ == "__main__":
    sys.exit(main())
