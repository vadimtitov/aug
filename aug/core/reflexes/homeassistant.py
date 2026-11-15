"""Home Assistant reflex: switches devices the moment the user asks, before the agent.

Two questions to OpenAI's Decisions API (aug/utils/decisions.py):

1. Is this message a request to operate a home device now?  Asked of every message, so
   it is one cheap yes/no; anything below GATE_THRESHOLD ends the reflex.
2. Only after a yes: per device the user labelled for AUG in HA, does the request
   target it, what should happen to it, and with what brightness, colour, white tone,
   position or temperature.  One request, all devices at once.

The answers become service calls only when every one of them is clear.  A device whose
targeting is neither a clear yes nor a clear no, an unsure parameter, an action the
device can't take: any of these and the reflex does nothing, and the agent handles the
message as it would anyway.  Doing nothing is always the safe outcome.

Credentials: HA_URL / HASS_TOKEN (see aug/config.py); the Decisions API through the
LiteLLM proxy (LLM_BASE_URL / LLM_API_KEY).  Devices: the entities carrying the label in
settings.json (reflexes → homeassistant → entity_label, default "aug"); their HA
aliases are given to the model as other names for them.
"""

import logging
import random
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
    HA_REFLEX_POSITION,
    HA_REFLEX_RELATIVE_BRIGHTNESS,
    HA_REFLEX_TARGET,
    HA_REFLEX_TEMPERATURE,
    HA_REFLEX_TONES,
)
from aug.core.reflexes import Exchange, ReflexOutput
from aug.utils.decisions import Answer, Choice, DecisionsClient, InputMessage, Predicate, Question
from aug.utils.file_settings import load_settings
from aug.utils.homeassistant import Entity, HomeAssistantClient

logger = logging.getLogger(__name__)

MODEL = "gpt-6-luna"
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
_BRIGHTNESSES = (1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
_BRIGHTNESS_STEP = {"brighter": 20, "dimmer": -20}
_POSITIONS = range(0, 101, 10)
_TEMPERATURES = range(16, 29)
_NEUTRAL_KELVIN = 4000
_COLOR_MODES = {"rgb", "rgbw", "rgbww", "hs", "xy"}
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

    async def __call__(self, query: str, history: Sequence[Exchange]) -> ReflexOutput | None:
        ha, decisions = self._clients()
        if ha is None or decisions is None:
            logger.info("ha_reflex_skip reason=not_configured")
            return None
        conversation = _conversation(query, history)
        gate = (await decisions.decide(conversation, [Predicate("gate", HA_REFLEX_GATE)]))["gate"]
        if gate.probability is None or gate.probability < GATE_THRESHOLD:
            logger.info("ha_reflex_skip reason=not_a_request p=%s", gate.probability)
            return None

        label = load_settings().reflexes.homeassistant.entity_label
        devices = [d for d in await ha.get_entities(label or None) if _questions(d, 0)]
        if not devices:
            logger.warning("ha_reflex_skip reason=no_devices label=%s", label or "<all>")
            return None
        questions = [q for i, d in enumerate(devices) for q in _questions(d, i)]
        catalog = InputMessage("developer", _catalog(devices))
        answers = await decisions.decide([catalog, *conversation], questions)
        calls = plan(devices, answers)
        if not calls:
            return None
        return await _execute(ha, calls)

    def _clients(self) -> tuple[HomeAssistantClient | None, DecisionsClient | None]:
        settings = get_settings()
        if not settings.ha_url or not settings.ha_token:
            return None, None
        if self._ha is None:
            self._ha = HomeAssistantClient(settings.ha_url, settings.ha_token)
        if self._decisions is None:
            self._decisions = DecisionsClient(
                settings.LLM_BASE_URL, settings.LLM_API_KEY, model=MODEL
            )
        return self._ha, self._decisions


def plan(devices: Sequence[Entity], answers: Mapping[str, Answer]) -> list[ServiceCall]:
    """The service calls *answers* add up to, or [] unless every answer is clear."""
    calls = []
    for i, device in enumerate(devices):
        targeted = answers[f"t{i}"].probability
        if targeted is None:
            return _unsure(device, "targeting refused")
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
            if answer.choice != _NOT_SPECIFIED:
                values[key] = answer.choice
        call = _call(device, values)
        if call is None:
            return _unsure(device, f"no service for {values}")
        calls.append(call)
    return calls


def _questions(device: Entity, i: int) -> list[Question]:
    """What layer 2 asks about *device*, its answers keyed by a letter and *i*.  [] for
    a device the reflex can't operate."""
    domain = device.domain
    actions = HA_REFLEX_ACTIONS.get(domain)
    if actions is None or device.state == "unavailable":
        return []
    f = {
        "name": device.friendly_name,
        "kind": HA_REFLEX_KINDS[domain],
        "area": device.area_name or "home",
        "aliases": _aliases(device),
    }
    questions: list[Question] = [
        Predicate(f"t{i}", HA_REFLEX_TARGET.format(**f)),
        Choice(f"a{i}", HA_REFLEX_ACTION.format(**f), actions),
    ]
    modes = set(device.attributes.get("supported_color_modes") or [])
    if domain == "light" and modes - {"onoff"}:
        options = {str(b): f"{b}%" for b in _BRIGHTNESSES} | HA_REFLEX_RELATIVE_BRIGHTNESS
        questions.append(_with_unspecified(f"b{i}", HA_REFLEX_BRIGHTNESS.format(**f), options))
    if domain == "light" and modes & _COLOR_MODES:
        questions.append(_with_unspecified(f"c{i}", HA_REFLEX_COLOR.format(**f), HA_REFLEX_COLORS))
    if domain == "light" and "color_temp" in modes:
        questions.append(
            _with_unspecified(f"k{i}", HA_REFLEX_COLOR_TEMP.format(**f), HA_REFLEX_TONES)
        )
    if domain == "cover":
        options = {str(p): f"{p}%" for p in _POSITIONS}
        questions.append(_with_unspecified(f"p{i}", HA_REFLEX_POSITION.format(**f), options))
    if domain == "climate":
        low = device.attributes.get("min_temp", _TEMPERATURES.start)
        high = device.attributes.get("max_temp", _TEMPERATURES.stop - 1)
        options = {str(t): f"{t} °C" for t in _TEMPERATURES if low <= t <= high}
        questions.append(_with_unspecified(f"h{i}", HA_REFLEX_TEMPERATURE.format(**f), options))
    return questions


def _with_unspecified(name: str, instructions: str, options: Mapping[str, str]) -> Choice:
    return Choice(name, instructions, {_NOT_SPECIFIED: HA_REFLEX_NOT_SPECIFIED, **options})


def _call(device: Entity, values: Mapping[str, str]) -> ServiceCall | None:
    """The one service call *values* (answers by question letter) describe for
    *device*, or None if they don't fit together."""
    domain, action = device.domain, values.get("a")
    params = {k: v for k, v in values.items() if k != "a"}
    data: dict[str, Any] = {}
    match domain, action:
        case "light", "turn_off" if not params:
            return ServiceCall("light.turn_off", device.entity_id, {})
        case "light", "turn_on":
            if "b" in params:
                b = params["b"]
                data |= (
                    {"brightness_step_pct": _BRIGHTNESS_STEP[b]}
                    if b in _BRIGHTNESS_STEP
                    else {"brightness_pct": int(b)}
                )
            if "c" in params:
                data["rgb_color"] = _RGB[params["c"]]
            if "k" in params:
                kelvin = _kelvin(device, params["k"])
                if kelvin is None:
                    return None
                data["color_temp_kelvin"] = kelvin
            if "c" in params and "k" in params:
                return None
            return ServiceCall("light.turn_on", device.entity_id, data)
        case "cover", _ if "p" in params and action in ("open", "close", "set_position"):
            return ServiceCall(
                "cover.set_cover_position", device.entity_id, {"position": int(params["p"])}
            )
        case "climate", "set_temperature" | "heat" if "h" in params:
            return ServiceCall(
                "climate.set_temperature", device.entity_id, {"temperature": int(params["h"])}
            )
        case "climate", "heat" | "turn_off" if not params:
            mode = "heat" if action == "heat" else "off"
            return ServiceCall("climate.set_hvac_mode", device.entity_id, {"hvac_mode": mode})
        case _, _ if not params and (domain, action) in _SERVICES:
            return ServiceCall(_SERVICES[(domain, action)], device.entity_id, {})
    return None


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


async def _execute(ha: HomeAssistantClient, calls: list[ServiceCall]) -> ReflexOutput | None:
    done: list[ServiceCall] = []
    failed: list[ServiceCall] = []
    for call in calls:
        try:
            await ha.call_service(call.service, call.entity_id, dict(call.data))
            logger.info("ha_action %s", call)
            done.append(call)
        except Exception:
            logger.exception("ha_action_failed %s", call)
            failed.append(call)
    if not done and not failed:
        return None
    lines = [f"- {c}" for c in done] + [f"- FAILED: {c}" for c in failed]
    note = "Home Assistant reflex:\n" + "\n".join(lines)
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
            kind=HA_REFLEX_KINDS[d.domain],
            area=d.area_name or "no room",
            aliases=_aliases(d),
        )
        for d in devices
    ]
    return HA_REFLEX_DEVICES.format(devices="\n".join(lines))


def _aliases(device: Entity) -> str:
    if not device.aliases:
        return ""
    return HA_REFLEX_ALIASES.format(aliases=", ".join(f'"{a}"' for a in device.aliases))


def _unsure(device: Entity, why: str) -> list[ServiceCall]:
    logger.info("ha_reflex_skip reason=unsure entity=%s %s", device.entity_id, why)
    return []
