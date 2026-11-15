"""Home Assistant reflex: switches devices the moment the user asks, before the agent.

Two questions to OpenAI's Decisions API (aug/utils/decisions.py):

1. Is this message a request to operate a home device now?  Asked of every message, so
   it is one cheap yes/no; anything below GATE_THRESHOLD ends the reflex.
2. Only after a yes: per device the user labelled for AUG in HA, does the request
   target it, what should happen to it, and with what brightness, colour, white tone,
   position or temperature.  One request, all devices at once.

The answers become service calls only when every one of them is clear.  A device whose
targeting is neither a clear yes nor a clear no, an unsure parameter, a value not on
offer, something the device can't do: any of these and the reflex does nothing, and the
agent handles the message as it would anyway.  Doing nothing is always the safe outcome.

Deciding is bounded by DECIDE_SECONDS.  Once the calls are decided they all go out, and
the note says what was done and what failed, so the agent never repeats or misreports
an action.  Only lights, switches, curtains and similar covers (never a garage door or
a gate), heating thermostats and media players are operated.

Credentials: HA_URL / HASS_TOKEN (see aug/config.py); the Decisions API through the
LiteLLM proxy (LLM_BASE_URL / LLM_API_KEY).  Devices: the entities carrying the label in
settings.json (reflexes → homeassistant → entity_label, default "aug"); their HA
aliases are given to the model as other names for them.
"""

import asyncio
import logging
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aug.config import get_settings
from aug.core.prompts import (
    HA_REFLEX_ACTION,
    HA_REFLEX_ACTIONS,
    HA_REFLEX_ALIASES,
    HA_REFLEX_BRIGHTNESS,
    HA_REFLEX_COLOR,
    HA_REFLEX_COLOR_TEMP,
    HA_REFLEX_COLORS,
    HA_REFLEX_DEVICE_LINE,
    HA_REFLEX_DEVICES,
    HA_REFLEX_GATE,
    HA_REFLEX_KINDS,
    HA_REFLEX_NOT_SPECIFIED,
    HA_REFLEX_OTHER,
    HA_REFLEX_POSITION,
    HA_REFLEX_RELATIVE_BRIGHTNESS,
    HA_REFLEX_TARGET,
    HA_REFLEX_TEMPERATURE,
    HA_REFLEX_TONES,
    HA_REFLEX_WHERE,
)
from aug.core.reflexes import Exchange, ReflexOutput
from aug.utils.decisions import Answer, Choice, DecisionsClient, InputMessage, Predicate, Question
from aug.utils.file_settings import load_settings
from aug.utils.homeassistant import Entity, HomeAssistantClient

logger = logging.getLogger(__name__)

MODEL = "gpt-6-luna"
# How long both questions together may take; past it the reflex gives up.  The first
# call after a quiet spell can take a couple of seconds.  Leaves the service calls room
# inside REFLEX_TIMEOUT_SECONDS.
DECIDE_SECONDS = 5.0
# The Decisions API's limit per request; a home with more questions than this is not
# asked at all.
MAX_QUESTIONS = 400
# Layer 1: how sure the model must be that the message is a device request.
GATE_THRESHOLD = 0.7
# Layer 2: a device is targeted at TARGET_THRESHOLD or above and left alone below
# IGNORE_THRESHOLD; anything between is too unclear to act on.
TARGET_THRESHOLD = 0.7
IGNORE_THRESHOLD = 0.3
# How sure an action or parameter must be.
CHOICE_CONFIDENCE = 0.5
# Earlier messages shown to the model, and how much of each.
HISTORY_MESSAGES = 2
HISTORY_CHARS = 300

_NOT_SPECIFIED = "not_specified"
_OTHER = "other"
_BRIGHTNESSES = (1, *range(5, 101, 5))
_BRIGHTNESS_STEP = {"brighter": 20, "dimmer": -20}
_POSITIONS = range(0, 101, 10)
_TEMPERATURES = range(16, 29)
_NEUTRAL_KELVIN = 4000
_COLOR_MODES = {"rgb", "rgbw", "rgbww", "hs", "xy"}
_COVER_SET_POSITION = 4  # CoverEntityFeature.SET_POSITION
_RGB = {
    "red": [255, 0, 0],
    "orange": [255, 128, 0],
    "yellow": [255, 220, 0],
    "green": [0, 255, 0],
    "cyan": [0, 255, 255],
    "blue": [0, 0, 255],
    "purple": [160, 32, 240],
    "pink": [255, 105, 180],
    "white": [255, 255, 255],
}
_SERVICES = {
    ("switch", "turn_on"): "switch.turn_on",
    ("switch", "turn_off"): "switch.turn_off",
    ("cover", "open"): "cover.open_cover",
    ("cover", "close"): "cover.close_cover",
    ("cover", "stop"): "cover.stop_cover",
    ("media_player", "turn_on"): "media_player.turn_on",
    ("media_player", "turn_off"): "media_player.turn_off",
    ("media_player", "play"): "media_player.media_play",
    ("media_player", "pause"): "media_player.media_pause",
    ("media_player", "volume_up"): "media_player.volume_up",
    ("media_player", "volume_down"): "media_player.volume_down",
}

_DISPLAY_PHRASES = [
    "Flipping switches...",
    "Telling your home what to do.",
    "Negotiating with your house.",
    "Your house complies.",
    "Bossing your devices around.",
    "Poking the smart home...",
    "Whispering to your devices...",
    "Pulling levers behind the scenes...",
    "Waking up your house...",
    "Convincing your home to cooperate...",
    "Commanding the castle.",
    "Herding your smart devices...",
    "Persuading your home...",
    "Reminding your home who's in charge.",
    "Overriding your home's free will...",
    "Sending your home a strongly-worded message.",
    "Putting your home to work.",
    "Summoning the home spirits...",
    "Wrangling your smart home...",
    "Pulling strings at home base...",
    "Reaching into the walls...",
    "Running home sorcery...",
    "Enforcing your will upon the house.",
    "Prodding your castle into action.",
    "Channeling your inner home overlord.",
    "Flexing some home control.",
    "Talking your home through it...",
    "Making the house dance.",
    "Dispatching orders to the castle.",
    "Nudging the machinery...",
    "Adjusting your domain.",
    "Exerting dominion over your home.",
    "Tickling the home network...",
    "Wrestling with your smart home.",
    "Taming the house.",
    "Doing home things.",
    "Manipulating your domestic environment.",
    "Running home witchcraft...",
    "Instructing your home in no uncertain terms.",
    "Making your dwelling comply.",
    "Asserting control over the household.",
    "Bothering your home network...",
    "Having a word with your house.",
    "Telling the house what's what.",
    "Poking things until they move.",
    "Nudging your home into submission.",
    "Running home errands you didn't have to lift a finger for.",
    "Delegating to the house.",
    "Making your home earn its keep.",
]


@dataclass(frozen=True)
class ServiceCall:
    service: str  # "light.turn_on"
    entity_id: str
    data: Mapping[str, Any]

    def __str__(self) -> str:
        data = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"{self.service} on {self.entity_id}" + (f" ({data})" if data else "")


class HomeAssistantReflex:
    """The reflex: an instance is the Reflex callable a profile lists."""

    __name__ = "homeassistant_reflex"

    def __init__(self) -> None:
        self._ha: HomeAssistantClient | None = None
        self._decisions: DecisionsClient | None = None
        self._clients_for: tuple[str, ...] | None = None

    async def __call__(self, query: str, history: Sequence[Exchange]) -> ReflexOutput | None:
        clients = self._clients()
        if clients is None:
            logger.info("ha_reflex_skip reason=not_configured")
            return None
        ha, decisions = clients
        try:
            async with asyncio.timeout(DECIDE_SECONDS):
                calls = await _decide(ha, decisions, query, history)
        except TimeoutError:
            logger.warning("ha_reflex_skip reason=decide_timeout after=%.1fs", DECIDE_SECONDS)
            return None
        return await _execute(ha, calls) if calls else None

    def _clients(self) -> tuple[HomeAssistantClient, DecisionsClient] | None:
        """The clients for the current settings; rebuilt when those change."""
        settings = get_settings()
        if not settings.ha_url or not settings.ha_token:
            return None
        key = (settings.ha_url, settings.ha_token, settings.LLM_BASE_URL, settings.LLM_API_KEY)
        if self._clients_for != key:
            self._ha = HomeAssistantClient(settings.ha_url, settings.ha_token)
            self._decisions = DecisionsClient(
                settings.LLM_BASE_URL, settings.LLM_API_KEY, model=MODEL
            )
            self._clients_for = key
        assert self._ha is not None and self._decisions is not None
        return self._ha, self._decisions


def plan(devices: Sequence[Entity], answers: Mapping[str, Answer]) -> list[ServiceCall]:
    """The service calls *answers* add up to, or [] unless every answer is clear."""
    calls = []
    for i, device in enumerate(devices):
        target = answers.get(f"t{i}")
        targeted = target.probability if target is not None else None
        if targeted is None:
            return _unsure(device, "targeting refused or missing")
        if targeted < IGNORE_THRESHOLD:
            continue
        if targeted < TARGET_THRESHOLD:
            return _unsure(device, f"targeted p={targeted:.2f}")
        values = {}
        for key in ("a", "b", "c", "k", "p", "h"):
            answer = answers.get(f"{key}{i}")
            if answer is None:
                continue
            if answer.choice is None or (answer.confidence or 0) < CHOICE_CONFIDENCE:
                return _unsure(device, f"question {key}")
            if answer.choice == _OTHER:
                return _unsure(device, f"question {key}: a value not on offer")
            if answer.choice != _NOT_SPECIFIED:
                values[key] = answer.choice
        call = _call(device, values)
        if call is None:
            return _unsure(device, f"no service for {values}")
        calls.append(call)
    return calls


def _questions(device: Entity, i: int) -> list[Question]:
    """What layer 2 asks about *device*, its answers keyed by a letter and *i*.  [] for
    a device the reflex doesn't operate.

    A light is asked about brightness, colour and white tone whatever it supports, so
    a request it can't carry out ("make the plain bulb red") is seen and refused rather
    than half done."""
    kind = _kind(device)
    if kind is None or device.state == "unavailable":
        return []
    f = {
        "name": device.friendly_name,
        "kind": kind,
        "where": _where(device),
        "aliases": _aliases(device),
    }
    questions: list[Question] = [
        Predicate(f"t{i}", HA_REFLEX_TARGET.format(**f)),
        Choice(f"a{i}", HA_REFLEX_ACTION.format(**f), HA_REFLEX_ACTIONS[device.domain]),
    ]
    match device.domain:
        case "light":
            brightness = {str(b): f"{b}%" for b in _BRIGHTNESSES} | HA_REFLEX_RELATIVE_BRIGHTNESS
            questions += [
                _parameter(f"b{i}", HA_REFLEX_BRIGHTNESS.format(**f), brightness),
                _parameter(f"c{i}", HA_REFLEX_COLOR.format(**f), HA_REFLEX_COLORS),
                _parameter(f"k{i}", HA_REFLEX_COLOR_TEMP.format(**f), HA_REFLEX_TONES),
            ]
        case "cover":
            positions = {str(p): f"{p}%" for p in _POSITIONS}
            questions.append(_parameter(f"p{i}", HA_REFLEX_POSITION.format(**f), positions))
        case "climate":
            low = device.attributes.get("min_temp", _TEMPERATURES.start)
            high = device.attributes.get("max_temp", _TEMPERATURES.stop - 1)
            degrees = {str(t): f"{t} °C" for t in _TEMPERATURES if low <= t <= high}
            questions.append(_parameter(f"h{i}", HA_REFLEX_TEMPERATURE.format(**f), degrees))
    return questions


def _kind(device: Entity) -> str | None:
    """What *device* is, in words; None for one the reflex doesn't operate."""
    domain = device.domain
    if domain == "climate" and "heat" not in (device.attributes.get("hvac_modes") or []):
        return None
    device_class = device.attributes.get("device_class")
    kind = HA_REFLEX_KINDS.get((domain, device_class))
    if kind is None and domain != "cover":  # an unknown cover could be a garage door
        kind = HA_REFLEX_KINDS.get((domain, None))
    return kind


def _parameter(name: str, instructions: str, options: Mapping[str, str]) -> Choice:
    return Choice(
        name,
        instructions,
        {_NOT_SPECIFIED: HA_REFLEX_NOT_SPECIFIED, **options, _OTHER: HA_REFLEX_OTHER},
    )


def _call(device: Entity, values: Mapping[str, str]) -> ServiceCall | None:
    """The one service call *values* (answers by question letter) describe for
    *device*, or None if they don't fit together or the device can't do it."""
    domain, action = device.domain, values.get("a")
    params = {k: v for k, v in values.items() if k != "a"}
    match domain, action:
        case "light", "turn_off" if not params:
            return ServiceCall("light.turn_off", device.entity_id, {})
        case "light", "turn_on":
            data = _light_data(device, params)
            return None if data is None else ServiceCall("light.turn_on", device.entity_id, data)
        case "cover", "open" | "close" | "set_position" if "p" in params:
            return _cover_position(device, action, int(params["p"]))
        case "climate", "heat" if "h" in params:
            data = {"temperature": int(params["h"]), "hvac_mode": "heat"}
            return ServiceCall("climate.set_temperature", device.entity_id, data)
        case "climate", "set_temperature" if "h" in params:
            data = {"temperature": int(params["h"])}
            return ServiceCall("climate.set_temperature", device.entity_id, data)
        case "climate", "heat" | "turn_off" if not params:
            mode = "heat" if action == "heat" else "off"
            return ServiceCall("climate.set_hvac_mode", device.entity_id, {"hvac_mode": mode})
        case _, _ if not params and (domain, action) in _SERVICES:
            return ServiceCall(_SERVICES[(domain, action)], device.entity_id, {})
    return None


def _light_data(device: Entity, params: Mapping[str, str]) -> dict[str, Any] | None:
    """light.turn_on's data for *params*, or None if the light can't do one of them."""
    modes = set(device.attributes.get("supported_color_modes") or [])
    data: dict[str, Any] = {}
    if "b" in params:
        if not modes - {"onoff"}:
            return None
        b = params["b"]
        if b in _BRIGHTNESS_STEP:
            data["brightness_step_pct"] = _BRIGHTNESS_STEP[b]
        else:
            data["brightness_pct"] = int(b)
    if "c" in params and "k" in params:
        return None
    if "c" in params:
        if not modes & _COLOR_MODES or params["c"] not in _RGB:
            return None
        data["rgb_color"] = _RGB[params["c"]]
    if "k" in params:
        kelvin = _kelvin(device, params["k"]) if "color_temp" in modes else None
        if kelvin is None:
            return None
        data["color_temp_kelvin"] = kelvin
    return data


def _cover_position(device: Entity, action: str, position: int) -> ServiceCall | None:
    if (action, position) in (("close", 100), ("open", 0)):
        return None  # the two answers disagree
    if position == 0:
        return ServiceCall("cover.close_cover", device.entity_id, {})
    if position == 100:
        return ServiceCall("cover.open_cover", device.entity_id, {})
    if not int(device.attributes.get("supported_features") or 0) & _COVER_SET_POSITION:
        return None
    return ServiceCall("cover.set_cover_position", device.entity_id, {"position": position})


def _kelvin(device: Entity, tone: str) -> int | None:
    """*tone* in kelvin for *device*.  None for a relative tone ('warmer'): how much is
    the agent's to judge."""
    low = device.attributes.get("min_color_temp_kelvin")
    high = device.attributes.get("max_color_temp_kelvin")
    if low is None or high is None:
        return None
    match tone:
        case "warmest":
            return int(low)
        case "coldest":
            return int(high)
        case "neutral":
            return int(min(max(_NEUTRAL_KELVIN, low), high))
    return None


async def _decide(
    ha: HomeAssistantClient, decisions: DecisionsClient, query: str, history: Sequence[Exchange]
) -> list[ServiceCall]:
    conversation = _conversation(query, history)
    label = load_settings().reflexes.homeassistant.entity_label
    # Read alongside layer 1 so it is off the path when the answer is yes; a read only,
    # and usually served from the client's cache.
    entities = asyncio.create_task(ha.get_entities(label or None))
    try:
        started = time.monotonic()
        gate = (await decisions.decide(conversation, [Predicate("gate", HA_REFLEX_GATE)]))["gate"]
        gate_ms = (time.monotonic() - started) * 1000
        if gate.probability is None or gate.probability < GATE_THRESHOLD:
            logger.info(
                "ha_reflex_skip reason=not_a_request p=%s gate_ms=%.0f", gate.probability, gate_ms
            )
            return []
        devices = [d for d in await entities if _questions(d, 0)]
    finally:
        entities.cancel()
        if entities.done() and not entities.cancelled():
            entities.exception()  # retrieved: a failed read only matters on a yes
    questions = [q for i, d in enumerate(devices) for q in _questions(d, i)]
    if not devices or len(questions) > MAX_QUESTIONS:
        logger.warning(
            "ha_reflex_skip reason=devices count=%d questions=%d label=%s",
            len(devices),
            len(questions),
            label or "<all>",
        )
        return []
    catalog = InputMessage("developer", _catalog(devices))
    started = time.monotonic()
    answers = await decisions.decide([catalog, *conversation], questions)
    logger.info(
        "ha_reflex_decided gate_ms=%.0f layer2_ms=%.0f questions=%d",
        gate_ms,
        (time.monotonic() - started) * 1000,
        len(questions),
    )
    return plan(devices, answers)


async def _execute(ha: HomeAssistantClient, calls: list[ServiceCall]) -> ReflexOutput:
    """Make every call at once; the note says which went through and which failed."""
    results = await asyncio.gather(
        *(ha.call_service(c.service, c.entity_id, dict(c.data)) for c in calls),
        return_exceptions=True,
    )
    done: list[ServiceCall] = []
    failed: list[ServiceCall] = []
    for call, result in zip(calls, results, strict=True):
        if isinstance(result, BaseException):
            logger.error("ha_action_failed %s: %r", call, result)
            failed.append(call)
        else:
            logger.info("ha_action %s", call)
            done.append(call)
    lines = [f"- done: {c}" for c in done] + [f"- FAILED: {c}" for c in failed]
    note = "Home Assistant:\n" + "\n".join(lines)
    display = None
    if done:
        partial = f" ({len(done)}/{len(calls)} succeeded)" if failed else ""
        display = f"🪄 {random.choice(_DISPLAY_PHRASES)}{partial}"
    return ReflexOutput(note=note, display=display)


def _conversation(query: str, history: Sequence[Exchange]) -> list[InputMessage]:
    earlier = [InputMessage(e.role, e.text[:HISTORY_CHARS]) for e in history[-HISTORY_MESSAGES:]]
    return [*earlier, InputMessage("user", query)]


def _catalog(devices: Sequence[Entity]) -> str:
    lines = [
        HA_REFLEX_DEVICE_LINE.format(
            name=d.friendly_name,
            kind=_kind(d),
            where=_where(d),
            aliases=_aliases(d),
        )
        for d in devices
    ]
    return HA_REFLEX_DEVICES.format(devices="\n".join(lines))


def _where(device: Entity) -> str:
    return HA_REFLEX_WHERE.format(area=device.area_name) if device.area_name else ""


def _aliases(device: Entity) -> str:
    if not device.aliases:
        return ""
    return HA_REFLEX_ALIASES.format(aliases=", ".join(f'"{a}"' for a in device.aliases))


def _unsure(device: Entity, why: str) -> list[ServiceCall]:
    logger.info("ha_reflex_skip reason=unsure entity=%s %s", device.entity_id, why)
    return []
