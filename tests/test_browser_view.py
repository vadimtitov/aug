"""Tests for aug.core.browser_view.BrowserViewHub.

Behaviors under test:
  - configured reflects whether a CDP URL is set; view() refuses when unset
  - the upstream screencast starts on the first viewer and stops on the last
  - frames fan out to every viewer
  - drop-to-latest: an unread frame is overwritten, viewers see only the newest
  - a new viewer is seeded immediately (from cached frame, or a fresh seed())
  - aclose() tears the screencast down
"""

import asyncio

from aug.core.browser_view import BrowserViewHub


class FakeScreencast:
    """Stand-in for BrowserScreencast — lets tests push frames synchronously."""

    def __init__(self, cdp_url, on_frame):
        self.cdp_url = cdp_url
        self.on_frame = on_frame
        self.started = False
        self.stopped = False
        self.seed_value: bytes | None = None

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def seed(self):
        return self.seed_value

    async def push(self, frame: bytes):
        await self.on_frame(frame)


def _hub_with_fake():
    created: list[FakeScreencast] = []

    def factory(cdp_url, on_frame):
        sc = FakeScreencast(cdp_url, on_frame)
        created.append(sc)
        return sc

    hub = BrowserViewHub("http://chromium:9222", screencast_factory=factory)
    return hub, created


async def test_unconfigured_hub_refuses_viewers():
    hub = BrowserViewHub(None)
    assert hub.configured is False
    try:
        async with hub.view():
            pass
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


async def test_lazy_start_and_stop():
    hub, created = _hub_with_fake()
    assert not created
    async with hub.view():
        assert len(created) == 1
        assert created[0].started is True
        assert created[0].stopped is False
    # last viewer left → screencast stopped
    assert created[0].stopped is True


async def test_single_screencast_shared_across_viewers():
    hub, created = _hub_with_fake()
    async with hub.view():
        async with hub.view():
            assert len(created) == 1  # one upstream, two viewers


async def test_frames_fan_out_to_all_viewers():
    hub, created = _hub_with_fake()
    async with hub.view() as v1, hub.view() as v2:
        await created[0].push(b"frame-1")
        assert await asyncio.wait_for(v1.get(), 1) == b"frame-1"
        assert await asyncio.wait_for(v2.get(), 1) == b"frame-1"


async def test_drop_to_latest():
    hub, created = _hub_with_fake()
    async with hub.view() as viewer:
        await created[0].push(b"old")
        await created[0].push(b"new")  # overwrites the unread frame
        assert await asyncio.wait_for(viewer.get(), 1) == b"new"


async def test_new_viewer_seeded_from_cached_frame():
    hub, created = _hub_with_fake()
    async with hub.view():
        await created[0].push(b"latest")
        # Second viewer joins after a frame already arrived → seeded immediately.
        async with hub.view() as v2:
            assert await asyncio.wait_for(v2.get(), 1) == b"latest"


async def test_new_viewer_seeded_via_seed_when_no_cache():
    hub, created = _hub_with_fake()

    def factory(cdp_url, on_frame):
        sc = FakeScreencast(cdp_url, on_frame)
        sc.seed_value = b"seeded"
        created.append(sc)
        return sc

    hub = BrowserViewHub("http://chromium:9222", screencast_factory=factory)
    async with hub.view() as viewer:
        assert await asyncio.wait_for(viewer.get(), 1) == b"seeded"


async def test_aclose_stops_screencast():
    hub, created = _hub_with_fake()
    viewer_cm = hub.view()
    await viewer_cm.__aenter__()
    assert created[0].stopped is False
    await hub.aclose()
    assert created[0].stopped is True
