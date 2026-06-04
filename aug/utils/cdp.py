"""CDP transport — talk to the remote Chromium over the DevTools Protocol.

Layer: client/transport ("how to talk to CDP"). The orchestration that decides
what to do with the frames (fan-out to viewers, lazy lifecycle) lives one layer
up in ``aug/core/browser_view.py``.

Chrome runs in a separate container, reachable over CDP. This module exposes:

  - ``resolve_cdp_url`` — rewrite a CDP URL's hostname to an IP so Chrome accepts
    the Host header (Chrome 94+ rejects non-IP/localhost Hosts on its CDP HTTP
    endpoint; inside Docker the host is a service name).
  - ``BrowserScreencast`` — connect to the browser, follow whichever tab the agent
    is currently driving, and emit live JPEG frames of it.

``BrowserScreencast`` owns its own reconnect loop: if Chrome restarts or the
socket drops, it retries with backoff and re-attaches transparently.
"""

import asyncio
import base64
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from cdp_use.client import CDPClient

logger = logging.getLogger(__name__)

FrameCallback = Callable[[bytes], Awaitable[None]]

# Screencast tuning. Hardcoded per the project's config philosophy — these are the
# established defaults for "watch the agent work" and there is no immediate reason
# for them to vary per user. Downscaling + JPEG quality keep mobile bandwidth sane;
# everyNthFrame caps the source frame rate so we never flood a slow link.
_FORMAT = "jpeg"
_QUALITY = 60
_MAX_WIDTH = 1280
_MAX_HEIGHT = 1280
_EVERY_NTH_FRAME = 1

_HTTP_TIMEOUT = 10.0
_RECONNECT_BACKOFF_START = 1.0
_RECONNECT_BACKOFF_MAX = 15.0


def is_ip_or_localhost(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False


def resolve_cdp_url(cdp_url: str) -> str:
    """Replace the hostname in a CDP URL with its resolved IP address.

    Chrome's CDP HTTP endpoint rejects requests whose Host header is not an IP or
    'localhost'. Inside Docker, service names resolve to container IPs, so we
    pre-resolve the hostname here to satisfy that check.
    """
    parsed = urlparse(cdp_url)
    if parsed.hostname and not is_ip_or_localhost(parsed.hostname):
        ip = socket.gethostbyname(parsed.hostname)
        resolved = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        return urlunparse(resolved)
    return cdp_url


class BrowserScreencast:
    """Stream JPEG frames of the page the agent is currently driving.

    Connects to the browser-level CDP endpoint and auto-attaches (flat mode) to
    every page target. Exactly one page — the "active" tab — is screencast at a
    time. The active tab is whichever one most recently appeared or navigated,
    which is exactly how an agent moves: it either navigates the current tab or
    opens a new one. When the active tab closes, we fall back to another open page.

    Frames are delivered to ``on_frame`` as raw JPEG bytes (already base64-decoded).
    """

    def __init__(self, cdp_url: str, on_frame: FrameCallback) -> None:
        self._cdp_url = cdp_url
        self._on_frame = on_frame
        self._client: CDPClient | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Strong refs to fire-and-forget handler tasks. asyncio only holds weak
        # references, so without this a task could be GC'd before it completes.
        self._jobs: set[asyncio.Task] = set()
        # session_id -> target_id for the page sessions we're attached to.
        self._pages: dict[str, str] = {}
        self._active: str | None = None
        self._latest: bytes | None = None

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("BrowserScreencast already started")
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for job in list(self._jobs):
            job.cancel()
        self._jobs.clear()

    async def seed(self) -> bytes | None:
        """Capture one immediate frame so a new viewer paints without waiting.

        ``startScreencast`` only emits on repaint, so an idle page would otherwise
        stay blank until something moves. Returns the last frame if we already have
        one, else a fresh ``captureScreenshot`` of the active tab, else ``None``.
        """
        if self._latest is not None:
            return self._latest
        client, active = self._client, self._active
        if client is None or active is None:
            return None
        try:
            result = await client.send.Page.captureScreenshot(
                params={"format": _FORMAT, "quality": _QUALITY}, session_id=active
            )
            return base64.b64decode(result["data"])
        except Exception as e:
            logger.debug("screencast seed failed: %s", e)
            return None

    # -- connection lifecycle -------------------------------------------------

    async def _run(self) -> None:
        backoff = _RECONNECT_BACKOFF_START
        first_failure = True
        while not self._stop.is_set():
            try:
                await self._connect_and_serve()
                backoff = _RECONNECT_BACKOFF_START
                first_failure = True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Warn once on the transition into a failing state, then stay quiet
                # at debug so a long Chrome outage doesn't spam the logs every 15s.
                log = logger.warning if first_failure else logger.debug
                log("screencast connection lost: %s; retrying in %.0fs", e, backoff)
                first_failure = False
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except TimeoutError:
                pass
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    async def _connect_and_serve(self) -> None:
        ws_url = await self._browser_ws_url()
        client = CDPClient(ws_url)
        await client.start()
        self._client = client
        self._pages.clear()
        self._active = None
        self._latest = None
        try:
            self._register_handlers(client)
            # Auto-attach (flat) to all current and future pages; events drive the rest.
            await client.send.Target.setAutoAttach(
                params={"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True}
            )
            # Block until the CDP socket closes, then raise to trigger a reconnect.
            # client.ws is the public websockets connection; wait_closed() is its
            # public API — we avoid reaching into cdp_use's private internals.
            if client.ws is not None:
                await client.ws.wait_closed()
            raise ConnectionError("CDP socket closed")
        finally:
            self._client = None
            self._pages.clear()
            self._active = None
            self._latest = None
            await client.stop()

    async def _browser_ws_url(self) -> str:
        resolved = resolve_cdp_url(self._cdp_url).rstrip("/")
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
            response = await http.get(f"{resolved}/json/version")
            response.raise_for_status()
            ws_url = response.json()["webSocketDebuggerUrl"]
        # Point the ws at the resolved host:port so it stays reachable via the proxy.
        base = urlparse(resolved)
        return urlunparse(urlparse(ws_url)._replace(netloc=base.netloc))

    # -- CDP event handlers ---------------------------------------------------
    #
    # These run inside CDPClient's receive loop, so they must NEVER await a CDP
    # command (its response is read by that same loop — awaiting would deadlock).
    # Each handler is a thin sync shim that schedules real work via create_task.

    def _register_handlers(self, client: CDPClient) -> None:
        client.register.Target.attachedToTarget(self._on_attached)
        client.register.Target.detachedFromTarget(self._on_detached)
        client.register.Page.frameNavigated(self._on_frame_navigated)
        client.register.Page.screencastFrame(self._on_screencast_frame)

    def _on_attached(self, event: dict[str, Any], _session_id: str | None) -> None:
        info = event.get("targetInfo", {})
        if info.get("type") != "page" or str(info.get("url", "")).startswith("devtools://"):
            return
        session = event["sessionId"]
        self._pages[session] = info.get("targetId", "")
        self._spawn(self._activate(session))

    def _on_detached(self, event: dict[str, Any], _session_id: str | None) -> None:
        session = event.get("sessionId")
        if session is None or session not in self._pages:
            return
        del self._pages[session]
        if session == self._active:
            self._active = None
            self._latest = None
            fallback = next(iter(self._pages), None)
            if fallback is not None:
                self._spawn(self._activate(fallback))

    def _on_frame_navigated(self, event: dict[str, Any], session_id: str | None) -> None:
        # Only top-frame navigations mark a tab as the one to watch (sub-frames have a parentId).
        if session_id is None or session_id not in self._pages:
            return
        if event.get("frame", {}).get("parentId"):
            return
        self._spawn(self._activate(session_id))

    def _on_screencast_frame(self, event: dict[str, Any], session_id: str | None) -> None:
        # Always ack so Chrome keeps producing; only forward frames of the active tab.
        # NB: CDP confusingly names the frame number "sessionId" on this event (and
        # on the ack params) — it is NOT the page session, which is `session_id`.
        client = self._client
        if client is not None:
            self._spawn(self._ack(client, session_id, event["sessionId"]))
        if session_id != self._active:
            return
        frame = base64.b64decode(event["data"])
        self._latest = frame
        self._spawn(self._on_frame(frame))

    # -- actions scheduled off the receive loop -------------------------------

    async def _activate(self, session: str) -> None:
        if session == self._active or session not in self._pages:
            return
        client = self._client
        if client is None:
            return
        previous = self._active
        self._active = session
        try:
            await client.send.Page.enable(session_id=session)
            await client.send.Page.startScreencast(
                params={
                    "format": _FORMAT,
                    "quality": _QUALITY,
                    "maxWidth": _MAX_WIDTH,
                    "maxHeight": _MAX_HEIGHT,
                    "everyNthFrame": _EVERY_NTH_FRAME,
                },
                session_id=session,
            )
        except Exception as e:
            logger.debug("startScreencast failed for %s: %s", session, e)
            return
        if previous is not None and previous in self._pages:
            try:
                await client.send.Page.stopScreencast(session_id=previous)
            except Exception:
                pass

    async def _ack(self, client: CDPClient, session: str | None, frame_number: int) -> None:
        # `frame_number` is the per-frame counter CDP labels "sessionId"; `session`
        # is the actual page session the ack is routed to.
        try:
            await client.send.Page.screencastFrameAck(
                params={"sessionId": frame_number}, session_id=session
            )
        except Exception:
            pass  # client tearing down or frame already superseded
