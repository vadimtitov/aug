"""Unit tests for the Home Assistant reflex."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

from aug.core.reflexes.homeassistant import (
    ALLOWED_DOMAINS,
    _credentials,
    _fetch_entities,
    _format_entities,
    _HAAction,
)
from aug.core.reflexes.homeassistant import (
    homeassistant_reflex as _homeassistant_reflex_factory,
)

homeassistant_reflex = _homeassistant_reflex_factory()

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_ENTITIES = [
    {
        "entity_id": "light.kitchen",
        "state": "off",
        "attributes": {"friendly_name": "Kitchen Light"},
    },
    {
        "entity_id": "switch.garden",
        "state": "on",
        "attributes": {"friendly_name": "Garden Switch"},
    },
    # Should be filtered out — not in ALLOWED_DOMAINS
    {
        "entity_id": "sensor.temperature",
        "state": "21.5",
        "attributes": {"friendly_name": "Temperature Sensor"},
    },
]


def _settings(ha_url=None, ha_token=None):
    m = MagicMock()
    m.ha_url = ha_url
    m.ha_token = ha_token
    return m


# ---------------------------------------------------------------------------
# _credentials
# ---------------------------------------------------------------------------


def test_credentials_returns_none_when_not_configured():
    with patch("aug.core.reflexes.homeassistant.get_settings", return_value=_settings()):
        assert _credentials() is None


def test_credentials_returns_none_when_token_missing():
    with patch(
        "aug.core.reflexes.homeassistant.get_settings",
        return_value=_settings(ha_url="http://ha:8123"),
    ):
        assert _credentials() is None


def test_credentials_returns_url_and_token_when_configured():
    with patch(
        "aug.core.reflexes.homeassistant.get_settings",
        return_value=_settings(ha_url="http://ha:8123", ha_token="tok"),
    ):
        assert _credentials() == ("http://ha:8123", "tok")


# ---------------------------------------------------------------------------
# _fetch_entities — filtering and caching
# ---------------------------------------------------------------------------


async def test_fetch_entities_filters_by_allowed_domain():
    import aug.core.reflexes.homeassistant as mod

    mod._entity_cache = []
    mod._entity_cache_at = 0.0

    mock_response = MagicMock()
    mock_response.json.return_value = _FAKE_ENTITIES
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("aug.core.reflexes.homeassistant.httpx.AsyncClient", return_value=mock_client):
        result = await _fetch_entities("http://ha:8123", "token")

    entity_ids = [e["entity_id"] for e in result]
    assert "light.kitchen" in entity_ids
    assert "switch.garden" in entity_ids
    assert "sensor.temperature" not in entity_ids


async def test_fetch_entities_uses_cache():
    import aug.core.reflexes.homeassistant as mod

    mod._entity_cache = [{"entity_id": "light.cached", "state": "on", "attributes": {}}]
    mod._entity_cache_at = time.monotonic()

    with patch("aug.core.reflexes.homeassistant.httpx.AsyncClient") as mock_cls:
        result = await _fetch_entities("http://ha:8123", "token")

    mock_cls.assert_not_called()
    assert result[0]["entity_id"] == "light.cached"


# ---------------------------------------------------------------------------
# _format_entities
# ---------------------------------------------------------------------------


def test_format_entities_includes_id_name_and_state():
    text = _format_entities(_FAKE_ENTITIES[:1])
    assert "light.kitchen" in text
    assert "Kitchen Light" in text
    assert "off" in text


# ---------------------------------------------------------------------------
# homeassistant_reflex — integration
# ---------------------------------------------------------------------------


async def test_reflex_returns_none_when_not_configured():
    with patch("aug.core.reflexes.homeassistant.get_settings", return_value=_settings()):
        assert await homeassistant_reflex("turn on lights", []) is None


async def test_reflex_returns_none_when_llm_finds_no_action():
    with (
        patch(
            "aug.core.reflexes.homeassistant.get_settings",
            return_value=_settings(ha_url="http://ha:8123", ha_token="tok"),
        ),
        patch(
            "aug.core.reflexes.homeassistant._fetch_entities",
            new=AsyncMock(return_value=_FAKE_ENTITIES),
        ),
        patch(
            "aug.core.reflexes.homeassistant._decide",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await homeassistant_reflex("are the lights on?", [])

    assert result is None


async def test_reflex_calls_service_and_returns_output():
    actions = [_HAAction(service="light.turn_on", entity_id="light.kitchen", service_data={})]
    mock_call = AsyncMock()

    with (
        patch(
            "aug.core.reflexes.homeassistant.get_settings",
            return_value=_settings(ha_url="http://ha:8123", ha_token="tok"),
        ),
        patch(
            "aug.core.reflexes.homeassistant._fetch_entities",
            new=AsyncMock(return_value=_FAKE_ENTITIES),
        ),
        patch("aug.core.reflexes.homeassistant._decide", new=AsyncMock(return_value=actions)),
        patch("aug.core.reflexes.homeassistant._call_service", new=mock_call),
    ):
        result = await homeassistant_reflex("turn on the kitchen light", [])

    assert result is not None
    assert "light.turn_on" in result.inject
    assert "light.kitchen" in result.inject
    assert result.display == "🏠 Home Assistant"
    mock_call.assert_awaited_once_with(
        "http://ha:8123", "tok", "light.turn_on", "light.kitchen", {}
    )


async def test_reflex_executes_multiple_actions():
    actions = [
        _HAAction(service="light.turn_off", entity_id="light.kitchen"),
        _HAAction(service="light.turn_off", entity_id="light.living_room"),
    ]
    mock_call = AsyncMock()

    with (
        patch(
            "aug.core.reflexes.homeassistant.get_settings",
            return_value=_settings(ha_url="http://ha:8123", ha_token="tok"),
        ),
        patch(
            "aug.core.reflexes.homeassistant._fetch_entities",
            new=AsyncMock(return_value=_FAKE_ENTITIES),
        ),
        patch("aug.core.reflexes.homeassistant._decide", new=AsyncMock(return_value=actions)),
        patch("aug.core.reflexes.homeassistant._call_service", new=mock_call),
    ):
        result = await homeassistant_reflex("turn off all lights", [])

    assert result is not None
    assert mock_call.await_count == 2
    assert "light.kitchen" in result.inject
    assert "light.living_room" in result.inject


async def test_reflex_returns_none_when_entities_empty():
    with (
        patch(
            "aug.core.reflexes.homeassistant.get_settings",
            return_value=_settings(ha_url="http://ha:8123", ha_token="tok"),
        ),
        patch(
            "aug.core.reflexes.homeassistant._fetch_entities",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await homeassistant_reflex("turn on lights", [])

    assert result is None


# ---------------------------------------------------------------------------
# ALLOWED_DOMAINS sanity check
# ---------------------------------------------------------------------------


def test_allowed_domains_contains_common_types():
    assert "light" in ALLOWED_DOMAINS
    assert "switch" in ALLOWED_DOMAINS
    assert "climate" in ALLOWED_DOMAINS
    assert "sensor" not in ALLOWED_DOMAINS
    assert "binary_sensor" not in ALLOWED_DOMAINS
