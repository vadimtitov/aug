"""Home Assistant reflex — executes unambiguous HA commands in parallel with the main agent.

Credentials are read from env vars (first match wins):
  URL:   HASS_URL, HA_URL, or HOMEASSISTANT_URL
  Token: HASS_TOKEN or HOMEASSISTANT_TOKEN

Entity label filter (settings.json: "reflexes" → "homeassistant" → "entity_label"):
  Default "aug" — only entities tagged with that label in HA are exposed to the LLM.
  Set to "" to expose all entities in ALLOWED_DOMAINS instead.

Model is configurable at registration time via homeassistant_reflex(model=...).
"""

import logging
import random

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from aug.config import get_settings
from aug.core.llm import build_chat_model
from aug.core.prompts import HA_REFLEX_SYSTEM_PROMPT
from aug.core.reflexes import Reflex, ReflexOutput
from aug.core.run import MessageContent
from aug.utils.file_settings import load_settings
from aug.utils.homeassistant import Entity, HomeAssistantClient

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"

_client: HomeAssistantClient | None = None

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


# ---------------------------------------------------------------------------
# Public reflex entry point
# ---------------------------------------------------------------------------


def homeassistant_reflex(model: str = _DEFAULT_MODEL) -> Reflex:
    """Return a reflex that executes unambiguous Home Assistant commands immediately."""

    async def _reflex(query: str, history: list[MessageContent]) -> ReflexOutput | None:
        client = _get_client()
        if client is None:
            logger.info("ha_reflex_skip reason=not_configured")
            return None

        label: str = load_settings().reflexes.homeassistant.entity_label
        entities = await client.get_entities(label or None)
        if not entities:
            logger.warning("ha_reflex_skip reason=no_entities label=%s", label or "<all>")
            return None

        actions = await _decide(query, _format_entities(entities), history, model)
        if not actions:
            logger.info("ha_reflex_skip reason=no_action_decided")
            return None

        executed: list[_HAAction] = []
        for action in actions:
            try:
                await client.call_service(action.service, action.entity_id, action.service_data)
                logger.info("ha_action service=%s entity_id=%s", action.service, action.entity_id)
                executed.append(action)
            except Exception:
                logger.exception(
                    "ha_action_failed service=%s entity_id=%s executed=%d/%d",
                    action.service,
                    action.entity_id,
                    len(executed),
                    len(actions),
                )
                break

        if not executed:
            return None

        lines = "\n".join(f"- {a.service} on {a.entity_id}" for a in executed)
        partial = (
            f" ({len(executed)}/{len(actions)} succeeded)" if len(executed) < len(actions) else ""
        )
        return ReflexOutput(
            inject=f"Home Assistant executed{partial}:\n{lines}",
            display=f"🪄 {random.choice(_DISPLAY_PHRASES)}{partial}",
        )

    _reflex.__name__ = "homeassistant_reflex"
    return _reflex


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


class _HAAction(BaseModel):
    service: str
    """HA service in domain.action format, e.g. 'light.turn_on'."""
    entity_id: str
    """Target entity ID, e.g. 'light.kitchen_ceiling'."""
    service_data: dict = Field(default_factory=dict)
    """Additional service parameters, e.g. {'temperature': 22} for climate."""


class _HADecision(BaseModel):
    actions: list[_HAAction] = Field(default_factory=list)
    """Empty list means the query is not an unambiguous HA command."""


def _get_client() -> HomeAssistantClient | None:
    global _client
    settings = get_settings()
    if not settings.ha_url or not settings.ha_token:
        return None
    if _client is None:
        _client = HomeAssistantClient(settings.ha_url, settings.ha_token)
    return _client


def _format_entities(entities: list[Entity]) -> str:
    lines = []
    for e in entities:
        location = f" in {e.area_name}" if e.area_name else ""
        lines.append(f"{e.entity_id} ({e.friendly_name}{location})")
    return "\n".join(lines)


async def _decide(
    query: str, entities_text: str, history: list[MessageContent], model: str
) -> list[_HAAction]:
    llm = build_chat_model(model=model, temperature=0).with_structured_output(
        _HADecision, method="function_calling"
    )
    history_text = (
        "\n\nRecent conversation:\n" + "\n".join(f"  {m}" for m in history if isinstance(m, str))
        if history
        else ""
    )
    messages = [
        SystemMessage(content=HA_REFLEX_SYSTEM_PROMPT),
        HumanMessage(content=f"Entities:\n{entities_text}{history_text}\n\nQuery: {query}"),
    ]
    result: _HADecision = await llm.ainvoke(messages)
    return result.actions
