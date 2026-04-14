"""Portainer REST API client."""

import logging

import httpx

from aug.config import get_settings

logger = logging.getLogger(__name__)


class PortainerClient:
    def __init__(self) -> None:
        s = get_settings()
        self._base = (s.PORTAINER_URL or "").rstrip("/")
        self._token = s.PORTAINER_API_TOKEN or ""

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._token}

    def is_configured(self) -> bool:
        return bool(self._base and self._token)

    async def list_endpoints(self) -> list[dict]:
        """Return all Portainer environments (endpoints)."""
        url = f"{self._base}/api/endpoints"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
        return r.json()

    async def resolve_endpoint(self, name: str) -> dict:
        """Return the endpoint dict for *name*, or raise ValueError listing available names."""
        endpoints = await self.list_endpoints()
        for e in endpoints:
            if e.get("Name") == name:
                return e
        available = ", ".join(e.get("Name", "?") for e in endpoints)
        raise ValueError(f"Environment {name!r} not found. Available: {available}")

    async def list_containers(self, endpoint_id: int) -> list[dict]:
        url = f"{self._base}/api/endpoints/{endpoint_id}/docker/containers/json"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=self._headers, params={"all": "1"})
            r.raise_for_status()
        return r.json()

    async def find_container_id(self, name_or_id: str, endpoint_id: int) -> str | None:
        """Resolve container name or partial ID to a full container ID."""
        containers = await self.list_containers(endpoint_id)
        target = name_or_id.lower()
        for c in containers:
            if c["Id"].startswith(target):
                return c["Id"]
            for n in c.get("Names", []):
                if n.lstrip("/").lower() == target:
                    return c["Id"]
        return None

    async def container_logs(self, container_id: str, endpoint_id: int, tail: int) -> bytes:
        url = f"{self._base}/api/endpoints/{endpoint_id}/docker/containers/{container_id}/logs"
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                url,
                headers=self._headers,
                params={"stdout": "1", "stderr": "1", "tail": str(tail)},
            )
            r.raise_for_status()
        return r.content

    async def container_action(self, container_id: str, endpoint_id: int, action: str) -> None:
        """Perform a lifecycle action on a container: start, stop, restart, remove."""
        if action == "remove":
            url = f"{self._base}/api/endpoints/{endpoint_id}/docker/containers/{container_id}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.delete(url, headers=self._headers)
                r.raise_for_status()
        else:
            url = (
                f"{self._base}/api/endpoints/{endpoint_id}"
                f"/docker/containers/{container_id}/{action}"
            )
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, headers=self._headers)
                r.raise_for_status()

    async def list_stacks(self) -> list[dict]:
        """Return all stacks across all endpoints."""
        url = f"{self._base}/api/stacks"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
        return r.json()

    async def find_stack(self, name: str, endpoint_id: int) -> dict | None:
        stacks = await self.list_stacks()
        return next(
            (s for s in stacks if s.get("Name") == name and s.get("EndpointId") == endpoint_id),
            None,
        )

    async def deploy_stack(self, name: str, compose: str, endpoint_id: int) -> tuple[dict, str]:
        """Create or update a stack. Returns (stack_data, action)."""
        existing = await self.find_stack(name, endpoint_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            if existing:
                url = f"{self._base}/api/stacks/{existing['Id']}"
                r = await client.put(
                    url,
                    headers=self._headers,
                    params={"endpointId": endpoint_id},
                    json={"stackFileContent": compose, "prune": True, "pullImage": True},
                )
                action = "updated"
            else:
                url = f"{self._base}/api/stacks/create/standalone/string"
                r = await client.post(
                    url,
                    headers=self._headers,
                    json={
                        "name": name,
                        "stackFileContent": compose,
                        "endpointId": endpoint_id,
                    },
                )
                action = "deployed"
            r.raise_for_status()
        return r.json(), action

    async def stack_action(self, stack_id: int, endpoint_id: int, action: str) -> None:
        """Perform a lifecycle action on a stack: start or stop."""
        url = f"{self._base}/api/stacks/{stack_id}/{action}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                url,
                headers=self._headers,
                params={"endpointId": endpoint_id},
            )
            r.raise_for_status()

    async def delete_stack(self, stack_id: int, endpoint_id: int) -> None:
        url = f"{self._base}/api/stacks/{stack_id}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.delete(
                url,
                headers=self._headers,
                params={"endpointId": endpoint_id},
            )
            r.raise_for_status()


def strip_docker_log_headers(data: bytes) -> str:
    """Strip 8-byte Docker multiplexing headers from log output."""
    lines: list[str] = []
    i = 0
    while i + 8 <= len(data):
        frame_size = int.from_bytes(data[i + 4 : i + 8], "big")
        end = i + 8 + frame_size
        chunk = data[i + 8 : end].decode("utf-8", errors="replace")
        lines.append(chunk.rstrip("\n"))
        i = end
    return "\n".join(lines) if lines else data.decode("utf-8", errors="replace")
