"""Unit tests for the Portainer tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_settings(url="http://portainer:9000", token="tok", endpoint_id=1):
    m = MagicMock()
    m.PORTAINER_URL = url
    m.PORTAINER_API_TOKEN = token
    m.PORTAINER_ENDPOINT_ID = endpoint_id
    return m


def _no_portainer():
    m = MagicMock()
    m.PORTAINER_URL = None
    m.PORTAINER_API_TOKEN = None
    m.PORTAINER_ENDPOINT_ID = 1
    return m


def _mock_client(json_data=None, status_code=200, content=b""):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = json_data or []
    mock_response.status_code = status_code
    mock_response.content = content

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# portainer_list_containers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_containers_not_configured():
    with patch("aug.utils.portainer.get_settings", return_value=_no_portainer()):
        from aug.core.tools.portainer import portainer_list_containers

        result = await portainer_list_containers.ainvoke({})
    assert "not configured" in result.lower()


@pytest.mark.asyncio
async def test_list_containers_success():
    containers = [
        {
            "Id": "abc123",
            "Names": ["/aug-aug-1"],
            "Image": "aug:latest",
            "State": "running",
            "Status": "Up 2 days",
        },
        {
            "Id": "def456",
            "Names": ["/postgres-1"],
            "Image": "postgres:15",
            "State": "running",
            "Status": "Up 2 days",
        },
    ]
    client = _mock_client(json_data=containers)
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_list_containers

        result = await portainer_list_containers.ainvoke({})

    assert "aug-aug-1" in result
    assert "postgres-1" in result
    assert "2 containers" in result


@pytest.mark.asyncio
async def test_list_containers_empty():
    client = _mock_client(json_data=[])
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_list_containers

        result = await portainer_list_containers.ainvoke({})

    assert "no containers" in result.lower()


# ---------------------------------------------------------------------------
# portainer_restart_container
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_not_configured():
    with patch("aug.utils.portainer.get_settings", return_value=_no_portainer()):
        from aug.core.tools.portainer import portainer_restart_container

        result = await portainer_restart_container.ainvoke({"container": "aug-aug-1"})
    assert "not configured" in result.lower()


@pytest.mark.asyncio
async def test_restart_container_not_found():
    client = _mock_client(json_data=[])  # empty list → not found
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_restart_container

        result = await portainer_restart_container.ainvoke({"container": "nonexistent"})
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_restart_container_success():
    containers = [
        {
            "Id": "abc123def456",
            "Names": ["/aug-aug-1"],
            "Image": "aug:latest",
            "State": "running",
            "Status": "Up",
        }
    ]
    client = _mock_client(json_data=containers)
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_restart_container

        result = await portainer_restart_container.ainvoke({"container": "aug-aug-1"})
    assert "restarted" in result.lower()


# ---------------------------------------------------------------------------
# portainer_list_stacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_stacks_success():
    stacks = [
        {"Id": 1, "Name": "aug", "Status": 1},
        {"Id": 2, "Name": "monitoring", "Status": 2},
    ]
    client = _mock_client(json_data=stacks)
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_list_stacks

        result = await portainer_list_stacks.ainvoke({})

    assert "aug" in result
    assert "monitoring" in result
    assert "active" in result
    assert "inactive" in result
