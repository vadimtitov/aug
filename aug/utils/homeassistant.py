"""Home Assistant async client.

Provides typed access to the HA REST and WebSocket APIs.
Entity lists are cached per label and refreshed after CACHE_TTL seconds.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
import websockets

logger = logging.getLogger(__name__)

CACHE_TTL = 300.0  # seconds


@dataclass(frozen=True)
class Entity:
    entity_id: str
    friendly_name: str
    state: str
    area_name: str | None = None

    @property
    def domain(self) -> str:
        return self.entity_id.split(".")[0]


class HomeAssistantClient:
    """Async client for the Home Assistant REST and WebSocket APIs.

    The entity cache is per-instance and per-label. Concurrent cache-miss
    requests are serialised by an asyncio.Lock so only one network round-trip
    is made regardless of how many coroutines call get_entities simultaneously.
    """

    def __init__(self, url: str, token: str, cache_ttl: float = CACHE_TTL) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._headers = {"Authorization": f"Bearer {token}"}
        self._cache_ttl = cache_ttl
        self._cache: list[Entity] = []
        self._cache_label: str | None = None
        self._cache_at: float = 0.0
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    async def get_entities(self, label: str | None = None) -> list[Entity]:
        """Return entities from Home Assistant.

        When *label* is given, only entities tagged with that label in the HA
        entity registry are returned. Otherwise all entities are returned via
        the REST states endpoint.

        Results are cached for cache_ttl seconds. The cache is invalidated
        when the requested label differs from the cached one.
        """
        async with self._lock:
            if self._is_cached(label):
                return self._cache
            entities = await self._fetch_by_label(label) if label else await self._fetch_all()
            self._cache = entities
            self._cache_label = label
            self._cache_at = time.monotonic()
            logger.debug("ha_entities_refreshed label=%s count=%d", label or "<all>", len(entities))
            return entities

    async def call_service(
        self, service: str, entity_id: str, service_data: dict | None = None
    ) -> None:
        """Call a HA service, e.g. ``light.turn_on``."""
        domain, action = service.split(".", 1)
        # Guard: entity_id must never appear in service_data — it would silently
        # override the top-level entity_id in the JSON payload.
        payload = {
            "entity_id": entity_id,
            **{k: v for k, v in (service_data or {}).items() if k != "entity_id"},
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._url}/api/services/{domain}/{action}",
                headers={**self._headers, "Content-Type": "application/json"},
                json=payload,
                timeout=5.0,
            )
            if response.is_error:
                logger.warning(
                    "ha_service_error service=%s status=%d body=%s",
                    service,
                    response.status_code,
                    response.text[:200],
                )
            response.raise_for_status()

    # ---------------------------------------------------------------------------
    # Private
    # ---------------------------------------------------------------------------

    def _is_cached(self, label: str | None) -> bool:
        return (
            bool(self._cache)
            and self._cache_label == label
            and time.monotonic() - self._cache_at < self._cache_ttl
        )

    async def _fetch_all(self) -> list[Entity]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._url}/api/states", headers=self._headers, timeout=5.0
            )
            response.raise_for_status()
        return [
            Entity(
                entity_id=s["entity_id"],
                friendly_name=s.get("attributes", {}).get("friendly_name") or s["entity_id"],
                state=s.get("state", "unknown"),
            )
            for s in response.json()
            if s.get("entity_id")
        ]

    async def _fetch_by_label(self, label: str) -> list[Entity]:
        matched_registry, device_area_map, area_map = await self._ws_fetch_registry(label)
        if not matched_registry:
            return []
        state_map = await self._fetch_states({e["entity_id"] for e in matched_registry})
        return [
            Entity(
                entity_id=e["entity_id"],
                friendly_name=(
                    state_map.get(e["entity_id"], {}).get("attributes", {}).get("friendly_name")
                    or e.get("name")
                    or e.get("original_name")
                    or e["entity_id"]
                ),
                state=state_map.get(e["entity_id"], {}).get("state", "unknown"),
                area_name=area_map.get(
                    e.get("area_id") or device_area_map.get(e.get("device_id", ""), "")
                ),
            )
            for e in matched_registry
        ]

    async def _ws_fetch_registry(
        self, label: str
    ) -> tuple[list[dict], dict[str, str], dict[str, str]]:
        """Return (label-matched entries, device_id→area_id map, area_id→name map) via WebSocket."""
        ws_url = (
            ("wss://" if self._url.startswith("https://") else "ws://")
            + self._url.split("://", 1)[1]
            + "/api/websocket"
        )
        async with websockets.connect(ws_url, max_size=16 * 1024 * 1024, open_timeout=5) as ws:
            await ws.recv()  # auth_required
            await ws.send(json.dumps({"type": "auth", "access_token": self._token}))
            auth = json.loads(await ws.recv())
            if auth["type"] != "auth_ok":
                raise RuntimeError(f"HA WebSocket auth failed: {auth}")

            await ws.send(json.dumps({"id": 1, "type": "config/entity_registry/list"}))
            await ws.send(json.dumps({"id": 2, "type": "config/area_registry/list"}))
            await ws.send(json.dumps({"id": 3, "type": "config/device_registry/list"}))

            responses: dict[int, list] = {}
            while len(responses) < 3:
                msg = json.loads(await ws.recv())
                if msg.get("id") not in (1, 2, 3):
                    continue  # skip unrelated events
                if not msg.get("success", True):
                    raise RuntimeError(f"HA WebSocket command {msg['id']} failed: {msg}")
                responses[msg["id"]] = msg.get("result", [])

        label_lower = label.lower()
        matched = [
            e for e in responses[1] if label_lower in [lb.lower() for lb in e.get("labels", [])]
        ]
        area_map = {a["area_id"]: a["name"] for a in responses[2]}
        device_area_map = {
            d["id"]: d["area_id"] for d in responses[3] if d.get("id") and d.get("area_id")
        }
        return matched, device_area_map, area_map

    async def _fetch_states(self, entity_ids: set[str]) -> dict[str, dict]:
        """Bulk-fetch current states for *entity_ids* via REST."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._url}/api/states", headers=self._headers, timeout=5.0
            )
            response.raise_for_status()
        return {s["entity_id"]: s for s in response.json() if s["entity_id"] in entity_ids}
