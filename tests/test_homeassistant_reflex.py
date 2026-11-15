"""The Home Assistant reflex (Decisions API answers → service calls) and the HA client."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.reflexes import Exchange
from aug.core.reflexes.homeassistant import (
    GATE_THRESHOLD,
    HomeAssistantReflex,
    ServiceCall,
    _questions,
    plan,
)
from aug.utils.decisions import Answer, Choice, DecisionsError, Predicate
from aug.utils.homeassistant import Entity, HomeAssistantClient

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _state(entity_id: str, state: str, friendly_name: str, **attributes) -> dict:
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": {"friendly_name": friendly_name, **attributes},
    }


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
    registry: list[dict],
    areas: list[dict],
    devices: list[dict] | None = None,
    entries: dict | None = None,
    entries_ok: bool = True,
) -> MagicMock:
    messages = [
        '{"type": "auth_required"}',
        '{"type": "auth_ok"}',
        json.dumps({"id": 1, "success": True, "result": registry}),
        json.dumps({"id": 2, "success": True, "result": areas}),
        json.dumps({"id": 3, "success": True, "result": devices or []}),
        json.dumps({"id": 4, "success": entries_ok, "result": entries or {}}),
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


async def test_get_entities_by_label_carries_aliases_and_attributes():
    client = HomeAssistantClient("http://ha:8123", "tok")
    registry = [_reg_entry("light.lamp", ["aug"])]
    entries = {"light.lamp": {"aliases": ["Лампа", None]}}
    states = [_state("light.lamp", "on", "Floor Lamp", supported_color_modes=["rgb"])]

    with (
        patch(
            "aug.utils.homeassistant.websockets.connect",
            return_value=_mock_ws(registry, [], entries=entries),
        ),
        patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=_mock_http_client(states)),
    ):
        (lamp,) = await client.get_entities(label="aug")

    assert lamp.aliases == ("Лампа",)
    assert lamp.attributes["supported_color_modes"] == ["rgb"]


async def test_entities_still_load_when_their_aliases_cannot_be_read():
    client = HomeAssistantClient("http://ha:8123", "tok")
    registry = [_reg_entry("light.lamp", ["aug"])]
    states = [_state("light.lamp", "on", "Floor Lamp")]

    with (
        patch(
            "aug.utils.homeassistant.websockets.connect",
            return_value=_mock_ws(registry, [], entries_ok=False),
        ),
        patch("aug.utils.homeassistant.httpx.AsyncClient", return_value=_mock_http_client(states)),
    ):
        (lamp,) = await client.get_entities(label="aug")

    assert lamp.aliases == ()


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
# The reflex: devices
# ---------------------------------------------------------------------------

LAMP = Entity(
    "light.lamp",
    "Floor Lamp",
    "on",
    area_name="Living Room",
    aliases=("Лампа",),
    attributes={
        "supported_color_modes": ["color_temp", "rgb"],
        "min_color_temp_kelvin": 2000,
        "max_color_temp_kelvin": 6500,
    },
)
SOFA = Entity(
    "light.sofa",
    "Sofa Light",
    "off",
    area_name="Living Room",
    attributes={"supported_color_modes": ["brightness"]},
)
PLUG = Entity("light.plug", "Plug Light", "off", attributes={"supported_color_modes": ["onoff"]})
CURTAIN = Entity(
    "cover.curtain",
    "Curtain",
    "open",
    area_name="Workroom",
    attributes={"device_class": "curtain", "supported_features": 15},
)
HEATING = Entity(
    "climate.thermostat",
    "Thermostat",
    "off",
    attributes={"hvac_modes": ["off", "heat"], "min_temp": 5, "max_temp": 30},
)
TV = Entity("media_player.tv", "TV", "off", area_name="Living Room")
SENSOR = Entity("sensor.temp", "Temp", "21")
GONE = Entity("switch.humidifier", "Humidifier", "unavailable")
GARAGE = Entity("cover.garage", "Garage", "closed", attributes={"device_class": "garage"})
UNKNOWN_COVER = Entity("cover.thing", "Thing", "closed")
AIRCON = Entity("climate.ac", "AC", "off", attributes={"hvac_modes": ["off", "cool"]})


def _names(device: Entity, i: int = 0) -> dict[str, list[str] | None]:
    return {
        q.name: (sorted(q.options) if isinstance(q, Choice) else None)
        for q in _questions(device, i)
    }


def test_each_device_gets_the_questions_of_its_kind():
    for light in (LAMP, SOFA, PLUG):  # every light, so a request it can't do is seen
        assert set(_names(light)) == {"t0", "a0", "b0", "c0", "k0"}
    assert set(_names(CURTAIN)) == {"t0", "a0", "p0"}
    assert set(_names(HEATING)) == {"t0", "a0", "h0"}
    assert set(_names(TV)) == {"t0", "a0"}


@pytest.mark.parametrize("device", [SENSOR, GONE, GARAGE, UNKNOWN_COVER, AIRCON])
def test_devices_the_reflex_must_not_operate_get_no_questions(device):
    assert _questions(device, 0) == []


def test_a_devices_aliases_and_room_are_part_of_its_question():
    target = _questions(LAMP, 3)[0]
    assert isinstance(target, Predicate) and target.name == "t3"
    assert "Floor Lamp" in target.instructions and "Лампа" in target.instructions
    assert "in the Living Room" in target.instructions
    assert " in the " not in _questions(PLUG, 0)[0].instructions


def test_temperatures_offered_stay_within_the_thermostats_range():
    narrow = Entity(
        "climate.t", "T", "off", attributes={"hvac_modes": ["heat"], "min_temp": 18, "max_temp": 20}
    )
    assert _names(narrow)["h0"] == sorted(["not_specified", "18", "19", "20", "other"])


# ---------------------------------------------------------------------------
# The reflex: answers -> service calls
# ---------------------------------------------------------------------------


def _target(p: float) -> Answer:
    return Answer(probability=p)


def _pick(value: str, confidence: float = 0.95) -> Answer:
    return Answer(choice=value, confidence=confidence)


def _unspecified(*names: str) -> dict[str, Answer]:
    return {n: _pick("not_specified") for n in names}


def _light(i: int, action: str = "turn_on", **params: str) -> dict[str, Answer]:
    answers = {f"t{i}": _target(0.95), f"a{i}": _pick(action)}
    for key in ("b", "c", "k"):
        answers[f"{key}{i}"] = _pick(params.get(key, "not_specified"))
    return answers


def test_one_message_can_set_two_devices_differently():
    # "turn the living room lamp red and the light to 20%"
    answers = {
        **_light(0, c="red"),
        **_light(1, b="20"),
        "t2": _target(0.02), "a2": _pick("open"), **_unspecified("p2"),
    }  # fmt: skip
    assert plan([LAMP, SOFA, CURTAIN], answers) == [
        ServiceCall("light.turn_on", "light.lamp", {"rgb_color": [255, 0, 0]}),
        ServiceCall("light.turn_on", "light.sofa", {"brightness_pct": 20}),
    ]


@pytest.mark.parametrize(
    ("device", "answers", "call"),
    [
        (SOFA, _light(0, "turn_off"), ServiceCall("light.turn_off", "light.sofa", {})),
        (
            SOFA,
            _light(0, b="brighter"),
            ServiceCall("light.turn_on", "light.sofa", {"brightness_step_pct": 20}),
        ),
        (
            LAMP,
            _light(0, k="warmest"),
            ServiceCall("light.turn_on", "light.lamp", {"color_temp_kelvin": 2000}),
        ),
        (PLUG, _light(0), ServiceCall("light.turn_on", "light.plug", {})),
        (
            CURTAIN,
            {"a0": _pick("close"), **_unspecified("p0")},
            ServiceCall("cover.close_cover", "cover.curtain", {}),
        ),
        (
            CURTAIN,
            {"a0": _pick("close"), "p0": _pick("50")},
            ServiceCall("cover.set_cover_position", "cover.curtain", {"position": 50}),
        ),
        (
            CURTAIN,
            {"a0": _pick("set_position"), "p0": _pick("0")},
            ServiceCall("cover.close_cover", "cover.curtain", {}),
        ),
        (
            HEATING,
            {"a0": _pick("set_temperature"), "h0": _pick("21")},
            ServiceCall("climate.set_temperature", "climate.thermostat", {"temperature": 21}),
        ),
        (
            HEATING,
            {"a0": _pick("heat"), "h0": _pick("22")},
            ServiceCall(
                "climate.set_temperature",
                "climate.thermostat",
                {"temperature": 22, "hvac_mode": "heat"},
            ),
        ),
        (
            HEATING,
            {"a0": _pick("turn_off"), **_unspecified("h0")},
            ServiceCall("climate.set_hvac_mode", "climate.thermostat", {"hvac_mode": "off"}),
        ),
        (
            TV,
            {"a0": _pick("pause")},
            ServiceCall("media_player.media_pause", "media_player.tv", {}),
        ),
    ],
)
def test_answers_become_the_matching_service_call(device, answers, call):
    assert plan([device], {"t0": _target(0.9), **answers}) == [call]


@pytest.mark.parametrize(
    ("device", "answers"),
    [
        # targeting neither clearly yes nor clearly no
        (SOFA, {**_light(0), "t0": _target(0.5)}),
        # unsure answers
        (SOFA, {**_light(0), "a0": _pick("turn_on", 0.4)}),
        (SOFA, {**_light(0), "b0": _pick("20", 0.3)}),
        # refused or missing answers
        (SOFA, {**_light(0), "a0": Answer()}),
        (SOFA, {**_light(0), "t0": Answer()}),
        (SOFA, {k: v for k, v in _light(0).items() if k != "t0"}),
        # a value not on offer ("35%", "21.5 degrees")
        (SOFA, _light(0, b="other")),
        (HEATING, {"t0": _target(0.9), "a0": _pick("set_temperature"), "h0": _pick("other")}),
        # parameters that contradict the action
        (SOFA, _light(0, "turn_off", b="50")),
        (LAMP, _light(0, c="red", k="warmest")),
        (CURTAIN, {"t0": _target(0.9), "a0": _pick("close"), "p0": _pick("100")}),
        # something the device can't do: must not be half done
        (PLUG, _light(0, b="20")),
        (PLUG, _light(0, c="red")),
        (SOFA, _light(0, c="red")),
        (SOFA, _light(0, k="warmest")),
        (
            Entity("cover.c", "C", "open", attributes={"device_class": "curtain"}),
            {"t0": _target(0.9), "a0": _pick("set_position"), "p0": _pick("50")},
        ),
        # relative white tones are the agent's to judge
        (LAMP, _light(0, k="warmer")),
        # a value outside what the reflex knows
        (LAMP, _light(0, c="ultraviolet")),
    ],
)
def test_anything_unclear_or_impossible_means_doing_nothing(device, answers):
    assert plan([device], answers) == []


def test_one_unclear_device_stops_the_whole_request():
    answers = {**_light(0), **_light(1), "t1": _target(0.4)}
    assert plan([SOFA, LAMP], answers) == []


# ---------------------------------------------------------------------------
# The reflex end to end (Decisions API and HA mocked)
# ---------------------------------------------------------------------------


class _FakeDecisions:
    """Answers the gate with *gate*, then layer 2 with *details*; records what it saw."""

    def __init__(self, gate: float | None, details: dict[str, Answer] | None = None, delay=0.0):
        self.gate = gate
        self.details = details or {}
        self.delay = delay
        self.calls: list[tuple[list, list]] = []

    async def decide(self, conversation, questions):
        self.calls.append((list(conversation), list(questions)))
        await asyncio.sleep(self.delay)
        if [q.name for q in questions] == ["gate"]:
            return {"gate": Answer(probability=self.gate)}
        return self.details


def _reflex(decisions, devices: list[Entity], fail: frozenset[str] = frozenset(), delay=0.0):
    ha = MagicMock()
    ha.get_entities = AsyncMock(return_value=devices)

    async def call_service(service, entity_id, data):
        await asyncio.sleep(delay)
        if entity_id in fail:
            raise RuntimeError("HA said no")

    ha.call_service = AsyncMock(side_effect=call_service)
    reflex = HomeAssistantReflex()
    reflex._clients = lambda: (ha, decisions)  # type: ignore[method-assign]
    return reflex, ha


_SOFA_TO_20 = _light(0, b="20")


@pytest.mark.parametrize("gate", [GATE_THRESHOLD - 0.01, None])
async def test_a_message_that_is_not_a_request_changes_nothing(gate):
    decisions = _FakeDecisions(gate=gate)
    reflex, ha = _reflex(decisions, [SOFA])

    assert await reflex("what's the weather?", []) is None
    ha.call_service.assert_not_awaited()
    assert len(decisions.calls) == 1


async def test_a_request_is_carried_out_and_reported():
    reflex, ha = _reflex(_FakeDecisions(gate=0.99, details=_SOFA_TO_20), [SOFA])

    output = await reflex("sofa light to 20%", [])

    ha.call_service.assert_awaited_once_with("light.turn_on", "light.sofa", {"brightness_pct": 20})
    assert output is not None and output.display
    assert "- done: light.turn_on on light.sofa (brightness_pct=20)" in output.note


async def test_a_failed_service_call_is_reported_as_failed():
    details = {**_SOFA_TO_20, **_light(1, "turn_off")}
    reflex, _ = _reflex(_FakeDecisions(0.99, details), [SOFA, LAMP], fail=frozenset({"light.sofa"}))

    output = await reflex("sofa to 20% and lamp off", [])

    assert output is not None and output.display and "(1/2 succeeded)" in output.display
    assert "- FAILED: light.turn_on on light.sofa" in output.note
    assert "- done: light.turn_off on light.lamp" in output.note


async def test_a_failed_device_read_does_not_matter_when_the_answer_is_no():
    reflex, ha = _reflex(_FakeDecisions(gate=0.1), [SOFA])
    ha.get_entities = AsyncMock(side_effect=RuntimeError("HA down"))
    assert await reflex("what's the weather?", []) is None


async def test_deciding_too_slowly_does_nothing():
    reflex, ha = _reflex(_FakeDecisions(0.99, _SOFA_TO_20, delay=0.2), [SOFA])
    with patch("aug.core.reflexes.homeassistant.DECIDE_SECONDS", 0.05):
        assert await reflex("sofa light to 20%", []) is None
    ha.call_service.assert_not_awaited()


async def test_once_decided_the_calls_finish_and_are_reported_whatever_the_deadline():
    reflex, ha = _reflex(_FakeDecisions(0.99, _SOFA_TO_20), [SOFA], delay=0.2)
    with patch("aug.core.reflexes.homeassistant.DECIDE_SECONDS", 0.05):
        output = await reflex("sofa light to 20%", [])
    ha.call_service.assert_awaited_once()
    assert output is not None and "- done:" in output.note


async def test_the_model_sees_recent_messages_and_the_devices():
    decisions = _FakeDecisions(gate=0.99, details={**_light(0), "t0": _target(0.01)})
    reflex, _ = _reflex(decisions, [SOFA])
    history = [
        Exchange("user", "old"),
        Exchange("user", "turn off the sofa light"),
        Exchange("assistant", "Done." + "x" * 1000),
    ]

    assert await reflex("now turn it on", history) is None

    gate_input, _ = decisions.calls[0]
    assert [(m.role, m.text[:23]) for m in gate_input] == [
        ("user", "turn off the sofa light"),
        ("assistant", "Done.xxxxxxxxxxxxxxxxxx"),
        ("user", "now turn it on"),
    ]
    assert len(gate_input[1].text) == 300
    detail_input, _ = decisions.calls[1]
    assert detail_input[0].role == "developer" and "Sofa Light" in detail_input[0].text


async def test_too_many_devices_to_ask_about_does_nothing():
    many = [Entity(f"light.l{i}", f"L{i}", "off") for i in range(100)]
    decisions = _FakeDecisions(gate=0.99)
    reflex, ha = _reflex(decisions, many)

    assert await reflex("turn on the lights", []) is None
    assert len(decisions.calls) == 1
    ha.call_service.assert_not_awaited()


async def test_a_decisions_api_failure_surfaces_to_run_reflexes():
    class _Down:
        async def decide(self, conversation, questions):
            raise DecisionsError("HTTP 503")

    reflex, _ = _reflex(_Down(), [SOFA])
    with pytest.raises(DecisionsError):
        await reflex("turn on the sofa light", [])


async def test_not_configured_does_nothing():
    with patch("aug.core.reflexes.homeassistant.get_settings") as settings:
        settings.return_value.ha_url = None
        settings.return_value.ha_token = None
        assert await HomeAssistantReflex()("turn on the light", []) is None


async def test_clients_follow_the_settings():
    reflex = HomeAssistantReflex()
    with patch("aug.core.reflexes.homeassistant.get_settings") as settings:
        settings.return_value.ha_url = "http://ha-1"
        settings.return_value.ha_token = "t"
        first = reflex._clients()
        assert reflex._clients() == first
        settings.return_value.ha_url = "http://ha-2"
        assert reflex._clients() != first
