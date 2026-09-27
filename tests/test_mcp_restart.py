"""Tests for aug/core/mcp_restart.py — restart delivery for MCP install/remove."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.mcp_restart import trigger_restart


def _settings(container="aug-aug-1", environment="musya"):
    m = MagicMock()
    m.AUG_CONTAINER = container
    m.AUG_ENVIRONMENT = environment
    return m


@pytest.mark.asyncio
async def test_trigger_restart_portainer_not_configured():
    update_calls = []
    mock_client = MagicMock()
    mock_client.is_configured.return_value = False

    with (
        patch("aug.core.mcp_restart.get_settings", return_value=_settings()),
        patch("aug.core.mcp_restart.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.mcp_restart.update_operation_state",
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
        ),
    ):
        result = await trigger_restart("op1", "postgres")

    assert "not configured" in result.lower()
    assert update_calls == [("op1", "restart_pending")]


@pytest.mark.asyncio
async def test_trigger_restart_container_or_environment_unset():
    """Even with Portainer itself configured, a missing AUG_CONTAINER/
    AUG_ENVIRONMENT must fall back to the manual-restart message rather than
    calling Portainer with an empty target."""
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True

    with (
        patch("aug.core.mcp_restart.get_settings", return_value=_settings(container=None)),
        patch("aug.core.mcp_restart.PortainerClient", return_value=mock_client),
        patch("aug.core.mcp_restart.update_operation_state", AsyncMock()),
    ):
        result = await trigger_restart("op1", "postgres")

    assert "not configured" in result.lower()
    mock_client.resolve_endpoint.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_restart_success_marks_pending_before_the_actual_restart_call():
    """The durable state must flip to restart_pending *before* the restart is
    actually requested — that request is what's expected to kill this very
    process moments later."""
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(return_value={"Id": 1})
    mock_client.find_container_id = AsyncMock(return_value="abc123")

    order = []
    mock_client.container_action = AsyncMock(side_effect=lambda *a: order.append("restart"))

    with (
        patch("aug.core.mcp_restart.get_settings", return_value=_settings()),
        patch("aug.core.mcp_restart.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.mcp_restart.update_operation_state",
            AsyncMock(side_effect=lambda *a: order.append(a)),
        ),
    ):
        result = await trigger_restart("op1", "postgres")

    assert "restart triggered" in result.lower()
    assert order == [("op1", "restart_pending"), "restart"]


@pytest.mark.asyncio
async def test_trigger_restart_container_not_found():
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(return_value={"Id": 1})
    mock_client.find_container_id = AsyncMock(return_value=None)

    update_calls = []
    with (
        patch("aug.core.mcp_restart.get_settings", return_value=_settings()),
        patch("aug.core.mcp_restart.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.mcp_restart.update_operation_state",
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
        ),
    ):
        result = await trigger_restart("op1", "postgres")

    assert "restart failed" in result.lower()
    assert update_calls[0][1] == "failed"


@pytest.mark.asyncio
async def test_trigger_restart_handles_exception():
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.resolve_endpoint = AsyncMock(side_effect=RuntimeError("portainer down"))

    update_calls = []
    with (
        patch("aug.core.mcp_restart.get_settings", return_value=_settings()),
        patch("aug.core.mcp_restart.PortainerClient", return_value=mock_client),
        patch(
            "aug.core.mcp_restart.update_operation_state",
            AsyncMock(side_effect=lambda *a: update_calls.append(a)),
        ),
    ):
        result = await trigger_restart("op1", "postgres")

    assert "restart failed" in result.lower()
    assert "portainer down" in result
    assert update_calls[0][1] == "failed"
