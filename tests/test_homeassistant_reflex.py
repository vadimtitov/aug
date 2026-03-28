"""Unit tests for the Home Assistant reflex and client."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aug.core.reflexes.homeassistant import (
    _decide,
    _format_entities,
    _HAAction,
)
from aug.core.reflexes.homeassistant import (
    homeassistant_reflex as _homeassistant_reflex_factory,
)
from aug.utils.homeassistant import Entity, HomeAssistantClient

homeassistant_reflex = _homeassistant_reflex_factory()

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _state(entity_id: str, state: str, friendly_name: str) -> dict:
    return {"entity_id": entity_id, "state": state, "attributes": {"friendly_name": friendly_name}}


_FAKE_STATES = [
    _state("light.kitchen", "off", "Kitchen Light"),
    _state("switch.garden", "on", "Garden Switch"),
    _state("sensor.temperature", "21.5", "Temp"),
    _state("humidifier.bedroom", "on", "Bedroom Humidifier"),
]

_FAKE_ENTITIES = [
    Entity(entity_id="light.kitchen", friendly_name="Kitchen Light", state="off"),
    Entity(
        entity_id="light.workroom", friendly_name="Workroom Light", state="on", area_name="Workroom"
    ),
]


def _reg_entry(
    entity_id: str,
    labels: list[str],
    area_id: str | None = None,
    original_name: str | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "labels": labels,
        "area_id": area_id,
        "name": None,
        "original_name": original_name,
        "device_id": None,
        "platform": entity_id.split(".")[0],
    }


def _mock_http_client(states: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = states
    response.raise_for_status = MagicMock()
    response.is_error = False
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    return client


def _mock_ws(
    registry: list[dict], areas: list[dict], devices: list[dict] | None = None
) -> MagicMock:
    messages = [
        '{"type": "auth_required"}',
        '{"type": "auth_ok"}',
        json.dumps({"id": 1, "success": True, "result": registry}),
        json.dumps({"id": 2, "success": True, "result": areas}),
        json.dumps({"id": 3, "success": True, "result": devices or []}),
    ]
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=False)
    ws.recv = AsyncMock(side_effect=messages)
    ws.send = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# HomeAssistantClient.get_entities — all domains
# ---------------------------------------------------------------------------


async def test_get_entities_all_returns_all_entities():
    client = HomeAssistantClient("http://ha:8123", "tok")
    mock = _mock_http_client(_FAKE_STATES)
    with patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=mock):
        entities = await client.get_entities()

    ids = [e.entity_id for e in entities]
    assert "light.kitchen" in ids
    assert "switch.garden" in ids
    assert "sensor.temperature" in ids  # all domains included now
    assert "humidifier.bedroom" in ids


async def test_get_entities_returns_cached_result():
    client = HomeAssistantClient("http://ha:8123", "tok")
    client._cache = _FAKE_ENTITIES
    client._cache_label = None
    client._cache_at = time.monotonic()

    with patch("aug.utils.homeassistant.httpx.AsyncClient") as mock_cls:
        await client.get_entities()

    mock_cls.assert_not_called()


async def test_get_entities_invalidates_cache_on_label_change():
    client = HomeAssistantClient("http://ha:8123", "tok")
    client._cache = _FAKE_ENTITIES
    client._cache_label = None  # cached for "all"
    client._cache_at = time.monotonic()

    registry = [_reg_entry("light.workroom", ["aug"], area_id="office", original_name="Workroom")]
    areas = [{"area_id": "office", "name": "Workroom"}]
    workroom_states = [_state("light.workroom", "on", "Workroom")]

    with (
        patch("aug.utils.homeassistant.websockets.connect", return_value=_mock_ws(registry, areas)),
        patch(
            "aug.utils.homeassistant.httpx.AsyncClient",
            return_value=_mock_http_client(workroom_states),
        ),
    ):
        result = await client.get_entities(label="aug")

    assert len(result) == 1
    assert result[0].entity_id == "light.workroom"


# ---------------------------------------------------------------------------
# HomeAssistantClient.get_entities — label-filtered
# ---------------------------------------------------------------------------


async def test_get_entities_by_label_filters_by_label_only():
    client = HomeAssistantClient("http://ha:8123", "tok")
    registry = [
        _reg_entry("light.kitchen", ["aug"], area_id="kitchen", original_name="Kitchen Light"),
        _reg_entry("sensor.temp", ["aug"]),  # included — no domain filter
        _reg_entry("light.bedroom", ["other"]),  # excluded — wrong label
    ]
    areas = [{"area_id": "kitchen", "name": "Kitchen"}]
    states = [
        _state("light.kitchen", "on", "Kitchen Light"),
        _state("sensor.temp", "21.5", "Temp"),
    ]

    with (
        patch("aug.utils.homeassistant.websockets.connect", return_value=_mock_ws(registry, areas)),
        patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=_mock_http_client(states)),
    ):
        result = await client.get_entities(label="aug")

    ids = [e.entity_id for e in result]
    assert len(result) == 2
    assert "light.kitchen" in ids
    assert "sensor.temp" in ids  # included now — user labeled it
    kitchen = next(e for e in result if e.entity_id == "light.kitchen")
    assert kitchen.area_name == "Kitchen"
    assert kitchen.state == "on"


async def test_get_entities_by_label_falls_back_to_device_area():
    """Area from device registry is used when entity has no area_id set directly."""
    client = HomeAssistantClient("http://ha:8123", "tok")
    registry = [_reg_entry("light.workroom", ["aug"], area_id=None, original_name="Workroom Light")]
    registry[0]["device_id"] = "dev-1"
    areas = [{"area_id": "workroom", "name": "Workroom"}]
    devices = [{"id": "dev-1", "area_id": "workroom"}]
    states = [_state("light.workroom", "on", "Workroom Light")]

    with (
        patch(
            "aug.utils.homeassistant.websockets.connect",
            return_value=_mock_ws(registry, areas, devices),
        ),
        patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=_mock_http_client(states)),
    ):
        result = await client.get_entities(label="aug")

    assert len(result) == 1
    assert result[0].area_name == "Workroom"


async def test_get_entities_by_label_returns_empty_when_no_match():
    client = HomeAssistantClient("http://ha:8123", "tok")
    registry = [_reg_entry("light.x", ["other"])]

    with patch("aug.utils.homeassistant.websockets.connect", return_value=_mock_ws(registry, [])):
        result = await client.get_entities(label="aug")

    assert result == []


# ---------------------------------------------------------------------------
# HomeAssistantClient.call_service
# ---------------------------------------------------------------------------


async def test_call_service_sends_correct_request():
    client = HomeAssistantClient("http://ha:8123", "tok")
    mock_http = _mock_http_client([])

    with patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=mock_http):
        await client.call_service("light.turn_on", "light.kitchen", {"brightness_pct": 80})

    mock_http.post.assert_awaited_once()
    _, kwargs = mock_http.post.call_args
    assert kwargs["json"] == {"entity_id": "light.kitchen", "brightness_pct": 80}


async def test_call_service_strips_entity_id_from_service_data():
    """LLM may hallucinate entity_id inside service_data — must not override the top-level one."""
    client = HomeAssistantClient("http://ha:8123", "tok")
    mock_http = _mock_http_client([])

    with patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=mock_http):
        await client.call_service(
            "light.turn_on", "light.kitchen", {"entity_id": "light.other", "brightness_pct": 50}
        )

    _, kwargs = mock_http.post.call_args
    assert kwargs["json"]["entity_id"] == "light.kitchen"
    assert "entity_id" not in {k for k in kwargs["json"] if k != "entity_id"}
    assert kwargs["json"]["brightness_pct"] == 50


# ---------------------------------------------------------------------------
# _format_entities
# ---------------------------------------------------------------------------


def test_format_entities_includes_id_and_name():
    text = _format_entities([Entity("light.kitchen", "Kitchen Light", "off")])
    assert "light.kitchen" in text
    assert "Kitchen Light" in text
    assert "off" not in text  # state omitted — cache may be stale


def test_format_entities_includes_area_when_present():
    text = _format_entities([Entity("light.workroom", "Workroom", "on", area_name="Workroom")])
    assert "in Workroom" in text


def test_format_entities_omits_area_when_absent():
    text = _format_entities([Entity("light.kitchen", "Kitchen", "off")])
    assert " in " not in text


# ---------------------------------------------------------------------------
# homeassistant_reflex — integration
# ---------------------------------------------------------------------------


async def test_reflex_returns_none_when_not_configured():
    with patch(
        "aug.core.reflexes.homeassistant.get_settings",
        return_value=MagicMock(ha_url=None, ha_token=None),
    ):
        assert await homeassistant_reflex("turn on lights", []) is None


async def test_reflex_returns_none_when_no_entities():
    mock_client = AsyncMock()
    mock_client.get_entities = AsyncMock(return_value=[])

    with (
        patch("aug.core.reflexes.homeassistant._get_client", return_value=mock_client),
        patch("aug.core.reflexes.homeassistant.get_setting", return_value="aug"),
    ):
        result = await homeassistant_reflex("turn on lights", [])

    assert result is None


async def test_reflex_returns_none_when_llm_finds_no_action():
    mock_client = AsyncMock()
    mock_client.get_entities = AsyncMock(return_value=_FAKE_ENTITIES)

    with (
        patch("aug.core.reflexes.homeassistant._get_client", return_value=mock_client),
        patch("aug.core.reflexes.homeassistant.get_setting", return_value="aug"),
        patch("aug.core.reflexes.homeassistant._decide", new=AsyncMock(return_value=[])),
    ):
        result = await homeassistant_reflex("are the lights on?", [])

    assert result is None


async def test_reflex_calls_service_and_returns_output():
    actions = [_HAAction(service="light.turn_on", entity_id="light.kitchen", service_data={})]
    mock_client = AsyncMock()
    mock_client.get_entities = AsyncMock(return_value=_FAKE_ENTITIES)
    mock_client.call_service = AsyncMock()

    with (
        patch("aug.core.reflexes.homeassistant._get_client", return_value=mock_client),
        patch("aug.core.reflexes.homeassistant.get_setting", return_value="aug"),
        patch("aug.core.reflexes.homeassistant._decide", new=AsyncMock(return_value=actions)),
    ):
        result = await homeassistant_reflex("turn on the kitchen light", [])

    assert result is not None
    assert "light.turn_on" in result.inject
    assert "light.kitchen" in result.inject
    assert result.display.startswith("🪄 ")
    mock_client.call_service.assert_awaited_once_with("light.turn_on", "light.kitchen", {})


async def test_reflex_reports_partial_failure():
    actions = [
        _HAAction(service="light.turn_off", entity_id="light.a"),
        _HAAction(service="light.turn_off", entity_id="light.b"),
    ]
    mock_client = AsyncMock()
    mock_client.get_entities = AsyncMock(return_value=_FAKE_ENTITIES)
    mock_client.call_service = AsyncMock(side_effect=[None, Exception("timeout")])

    with (
        patch("aug.core.reflexes.homeassistant._get_client", return_value=mock_client),
        patch("aug.core.reflexes.homeassistant.get_setting", return_value="aug"),
        patch("aug.core.reflexes.homeassistant._decide", new=AsyncMock(return_value=actions)),
    ):
        result = await homeassistant_reflex("turn off all lights", [])

    assert result is not None
    assert "1/2" in result.display
    assert "light.a" in result.inject
    assert "light.b" not in result.inject


# ---------------------------------------------------------------------------
# _decide — history in prompt
# ---------------------------------------------------------------------------


async def test_decide_includes_history_in_llm_prompt():
    captured: list = []
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        side_effect=lambda msgs: captured.extend(msgs) or MagicMock(actions=[])
    )
    history = ["User: turn on kitchen light", "Assistant: Done."]

    with patch(
        "aug.core.reflexes.homeassistant.build_chat_model",
        return_value=MagicMock(with_structured_output=MagicMock(return_value=mock_structured)),
    ):
        await _decide("now turn it off", "light.kitchen (Kitchen) [on]", history, "test-model")

    human = next(m for m in captured if hasattr(m, "content") and "Query" in m.content)
    assert "User: turn on kitchen light" in human.content
    assert "Assistant: Done." in human.content
    assert "now turn it off" in human.content


async def test_decide_omits_history_section_when_empty():
    captured: list = []
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        side_effect=lambda msgs: captured.extend(msgs) or MagicMock(actions=[])
    )

    with patch(
        "aug.core.reflexes.homeassistant.build_chat_model",
        return_value=MagicMock(with_structured_output=MagicMock(return_value=mock_structured)),
    ):
        await _decide("turn on kitchen", "light.kitchen (Kitchen) [off]", [], "test-model")

    human = next(m for m in captured if hasattr(m, "content") and "Query" in m.content)
    assert "Recent conversation" not in human.content
