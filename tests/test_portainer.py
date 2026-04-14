"""Unit tests for the Portainer tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_settings(url="http://portainer:9000", token="tok"):
    m = MagicMock()
    m.PORTAINER_URL = url
    m.PORTAINER_API_TOKEN = token
    return m


def _no_portainer():
    m = MagicMock()
    m.PORTAINER_URL = None
    m.PORTAINER_API_TOKEN = None
    return m


_ENDPOINTS = [
    {"Id": 1, "Name": "musya"},
    {"Id": 2, "Name": "dusya"},
]

_CONTAINERS = [
    {
        "Id": "abc123def456",
        "Names": ["/aug-aug-1"],
        "Image": "aug:latest",
        "State": "running",
        "Status": "Up 2 days",
    },
    {
        "Id": "def456abc123",
        "Names": ["/postgres-1"],
        "Image": "postgres:15",
        "State": "running",
        "Status": "Up 2 days",
    },
]


def _mock_client_multi(endpoints=None, containers=None, status_code=200, content=b""):
    """Build an AsyncClient mock: returns endpoints on GET /endpoints, containers otherwise."""
    endpoints = endpoints if endpoints is not None else _ENDPOINTS
    containers = containers if containers is not None else _CONTAINERS

    def _get_response(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.status_code = status_code
        resp.content = content
        if "/endpoints" in url and "/docker" not in url:
            resp.json.return_value = endpoints
        else:
            resp.json.return_value = containers
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=_get_response)
    mock_client.post = AsyncMock(return_value=_make_ok_response())
    mock_client.delete = AsyncMock(return_value=_make_ok_response())
    return mock_client


def _make_ok_response(json_data=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data or {}
    return resp


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
async def test_list_containers_grouped_by_environment():
    client = _mock_client_multi()
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_list_containers

        result = await portainer_list_containers.ainvoke({})

    assert "musya" in result
    assert "dusya" in result
    assert "aug-aug-1" in result


@pytest.mark.asyncio
async def test_list_containers_empty():
    client = _mock_client_multi(containers=[])
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_list_containers

        result = await portainer_list_containers.ainvoke({})

    assert "no containers" in result.lower()


# ---------------------------------------------------------------------------
# portainer_container_logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_logs_environment_not_found():
    client = _mock_client_multi()
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_container_logs

        result = await portainer_container_logs.ainvoke(
            {"container": "aug-aug-1", "environment": "nonexistent"}
        )
    assert "not found" in result.lower()
    assert "musya" in result
    assert "dusya" in result


@pytest.mark.asyncio
async def test_container_logs_container_not_found():
    client = _mock_client_multi(containers=[])
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_container_logs

        result = await portainer_container_logs.ainvoke(
            {"container": "nonexistent", "environment": "musya"}
        )
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# portainer_container_action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_action_not_configured():
    with patch("aug.utils.portainer.get_settings", return_value=_no_portainer()):
        from aug.core.tools.portainer import portainer_container_action

        result = await portainer_container_action.ainvoke(
            {"container": "aug-aug-1", "action": "restart", "environment": "musya"}
        )
    assert "not configured" in result.lower()


@pytest.mark.asyncio
async def test_container_action_success():
    client = _mock_client_multi()
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_container_action

        result = await portainer_container_action.ainvoke(
            {"container": "aug-aug-1", "action": "restart", "environment": "musya"}
        )
    assert "succeeded" in result.lower()
    assert "musya" in result


@pytest.mark.asyncio
async def test_container_action_not_found():
    client = _mock_client_multi(containers=[])
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_container_action

        result = await portainer_container_action.ainvoke(
            {"container": "ghost", "action": "stop", "environment": "dusya"}
        )
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# portainer_list_stacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_stacks_grouped_by_environment():
    stacks = [
        {"Id": 1, "Name": "aug", "Status": 1, "EndpointId": 1},
        {"Id": 2, "Name": "monitoring", "Status": 2, "EndpointId": 2},
    ]

    def _get_response(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.status_code = 200
        if "/endpoints" in url and "/docker" not in url:
            resp.json.return_value = _ENDPOINTS
        else:
            resp.json.return_value = stacks
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=_get_response)

    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=mock_client),
    ):
        from aug.core.tools.portainer import portainer_list_stacks

        result = await portainer_list_stacks.ainvoke({})

    assert "musya" in result
    assert "dusya" in result
    assert "aug" in result
    assert "monitoring" in result
    assert "active" in result
    assert "inactive" in result


# ---------------------------------------------------------------------------
# portainer_deploy_stack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_stack_environment_required():
    def _get_response(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.status_code = 200
        if "/endpoints" in url and "/docker" not in url:
            resp.json.return_value = _ENDPOINTS
        else:
            resp.json.return_value = []
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=_get_response)
    mock_client.post = AsyncMock(return_value=_make_ok_response({"Id": 42}))

    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=mock_client),
    ):
        from aug.core.tools.portainer import portainer_deploy_stack

        result = await portainer_deploy_stack.ainvoke(
            {"name": "myapp", "compose": "version: '3'", "environment": "musya"}
        )
    assert "deployed" in result.lower() or "updated" in result.lower()
    assert "musya" in result


@pytest.mark.asyncio
async def test_deploy_stack_unknown_environment():
    client = _mock_client_multi()
    with (
        patch("aug.utils.portainer.get_settings", return_value=_make_settings()),
        patch("aug.utils.portainer.httpx.AsyncClient", return_value=client),
    ):
        from aug.core.tools.portainer import portainer_deploy_stack

        result = await portainer_deploy_stack.ainvoke(
            {"name": "myapp", "compose": "version: '3'", "environment": "mars"}
        )
    assert "not found" in result.lower()
    assert "musya" in result
