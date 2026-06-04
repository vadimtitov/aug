"""Browser view hub — one live screencast fanned out to many viewers.

Orchestration layer over ``aug.utils.cdp.BrowserScreencast``. Responsibilities:

  - Lazy lifecycle: the upstream screencast (and the load it puts on Chrome) runs
    only while at least one viewer is watching. The last viewer leaving stops it.
  - Fan-out with **drop-to-latest**: each viewer holds a single-slot mailbox, so a
    slow client always jumps to the newest frame instead of accumulating lag — and
    can never back-pressure Chrome.
  - First-frame seeding: a newly attached viewer is handed the last frame (or a
    fresh screenshot) immediately, so it paints without waiting for a repaint.

The hub is interface-agnostic; the WebSocket router in ``aug.api.routers.browser``
is its only consumer today.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from aug.utils.cdp import BrowserScreencast, FrameCallback

logger = logging.getLogger(__name__)

ScreencastFactory = Callable[[str, FrameCallback], BrowserScreencast]


class Viewer:
    """Single-slot mailbox for one watching client.

    Overwriting an unread frame is the whole point: viewers always render the most
    recent frame, never a backlog. ``get`` returns ``None`` once the viewer is closed.
    """

    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._event = asyncio.Event()
        self._closed = False

    def put(self, frame: bytes) -> None:
        self._frame = frame
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def get(self) -> bytes | None:
        await self._event.wait()
        self._event.clear()
        if self._closed:
            return None
        frame, self._frame = self._frame, None
        return frame


class BrowserViewHub:
    """Manages a single upstream screencast and fans its frames out to viewers.

    There is one shared screencast for the whole process — every viewer watches
    the same browser. That matches AUG's single-user, single-browser model (one
    Chrome profile, one agent). It is deliberately NOT per-user isolated: the
    stream may show logged-in sessions or OTPs, so in a multi-tenant deployment
    this would need partitioning by the authenticated user.
    """

    def __init__(
        self,
        cdp_url: str | None,
        *,
        screencast_factory: ScreencastFactory = BrowserScreencast,
    ) -> None:
        self._cdp_url = cdp_url
        self._factory = screencast_factory
        self._viewers: set[Viewer] = set()
        self._screencast: BrowserScreencast | None = None
        self._latest: bytes | None = None
        self._lock = asyncio.Lock()
        # Strong refs to in-flight seed tasks (asyncio holds only weak refs).
        self._seed_tasks: set[asyncio.Task] = set()

    @property
    def configured(self) -> bool:
        return bool(self._cdp_url)

    @asynccontextmanager
    async def view(self) -> AsyncIterator[Viewer]:
        """Attach a viewer for the duration of the ``async with`` block."""
        viewer = await self._add()
        try:
            yield viewer
        finally:
            await self._remove(viewer)

    async def _add(self) -> Viewer:
        if not self._cdp_url:
            raise RuntimeError("Browser view is not configured (BROWSER_CDP_URL unset).")
        viewer = Viewer()
        async with self._lock:
            self._viewers.add(viewer)
            if self._screencast is None:
                self._screencast = self._factory(self._cdp_url, self._on_frame)
                await self._screencast.start()
                logger.info("browser_view: screencast started")
        if self._latest is not None:
            viewer.put(self._latest)
        else:
            task = asyncio.create_task(self._seed(self._screencast))
            self._seed_tasks.add(task)
            task.add_done_callback(self._seed_tasks.discard)
        return viewer

    async def _remove(self, viewer: Viewer) -> None:
        async with self._lock:
            self._viewers.discard(viewer)
            viewer.close()
            if not self._viewers and self._screencast is not None:
                screencast, self._screencast = self._screencast, None
                self._latest = None
                await screencast.stop()
                logger.info("browser_view: screencast stopped (no viewers)")

    async def _seed(self, screencast: BrowserScreencast) -> None:
        frame = await screencast.seed()
        if frame is None:
            return
        # Re-check under the lock: the viewer may have left and the screencast been
        # swapped while seed() awaited, in which case this frame is stale — dropping
        # it avoids pushing a dead frame to whatever viewer arrives next.
        async with self._lock:
            if self._screencast is screencast:
                self._dispatch(frame)

    async def _on_frame(self, frame: bytes) -> None:
        self._dispatch(frame)

    def _dispatch(self, frame: bytes) -> None:
        self._latest = frame
        for viewer in self._viewers:
            viewer.put(frame)

    async def aclose(self) -> None:
        """Tear down on application shutdown."""
        async with self._lock:
            for viewer in self._viewers:
                viewer.close()
            self._viewers.clear()
            if self._screencast is not None:
                screencast, self._screencast = self._screencast, None
                self._latest = None
                await screencast.stop()
