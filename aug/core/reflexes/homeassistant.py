"""Home Assistant reflex — executes unambiguous HA commands in parallel with the main agent.

Credentials are read from env vars (first match wins):
  URL:   HA_URL  or  HOMEASSISTANT_URL
  Token: HASS_TOKEN  or  HOMEASSISTANT_TOKEN

Entity list is fetched from HA on first use and cached for ENTITY_CACHE_TTL seconds.
Only entities in ALLOWED_DOMAINS are passed to the LLM to keep the context tight.

Model is configurable via user settings key ("reflexes", "homeassistant", "model").
"""

import asyncio
import logging
import time

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from aug.config import get_settings
from aug.core.llm import build_chat_model
from aug.core.prompts import HA_REFLEX_SYSTEM_PROMPT
from aug.core.reflexes import Reflex, ReflexOutput
from aug.core.run import MessageContent

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash-lite"
_ENTITY_CACHE_TTL = 300.0  # seconds

ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "light",
        "switch",
        "climate",
        "cover",
        "media_player",
        "fan",
        "scene",
        "script",
        "input_boolean",
        "lock",
        "vacuum",
    }
)

_entity_cache: list[dict] = []
_entity_cache_at: float = 0.0
_entity_cache_lock: asyncio.Lock | None = None


# ---------------------------------------------------------------------------
# Public reflex entry point
# ---------------------------------------------------------------------------


def homeassistant_reflex(model: str = _DEFAULT_MODEL) -> Reflex:
    """Return a reflex that executes unambiguous Home Assistant commands immediately."""

    async def _reflex(query: str, history: list[MessageContent]) -> ReflexOutput | None:
        creds = _credentials()
        if creds is None:
            logger.info("ha_reflex_skip reason=not_configured")
            return None

        url, token = creds
        entities = await _fetch_entities(url, token)
        if not entities:
            logger.warning("ha_reflex_skip reason=no_entities")
            return None

        actions = await _decide(query, _format_entities(entities), history, model)
        if not actions:
            logger.info("ha_reflex_skip reason=no_action_decided")
            return None

        executed: list[_HAAction] = []
        for action in actions:
            try:
                await _call_service(
                    url, token, action.service, action.entity_id, action.service_data
                )
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
            display=f"🏠 Home Assistant{partial}",
        )

    _reflex.__name__ = "homeassistant_reflex"
    return _reflex


# ---------------------------------------------------------------------------
# Private helpers
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


def _credentials() -> tuple[str, str] | None:
    settings = get_settings()
    if settings.ha_url and settings.ha_token:
        return settings.ha_url, settings.ha_token
    return None


async def _fetch_entities(url: str, token: str) -> list[dict]:
    global _entity_cache, _entity_cache_at, _entity_cache_lock
    if _entity_cache and time.monotonic() - _entity_cache_at < _ENTITY_CACHE_TTL:
        return _entity_cache
    if _entity_cache_lock is None:
        _entity_cache_lock = asyncio.Lock()
    async with _entity_cache_lock:
        # Re-check inside the lock — another task may have populated it while we waited.
        if _entity_cache and time.monotonic() - _entity_cache_at < _ENTITY_CACHE_TTL:
            return _entity_cache
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/api/states",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            )
            response.raise_for_status()
        states: list[dict] = response.json()
        filtered = [s for s in states if s.get("entity_id", "").split(".")[0] in ALLOWED_DOMAINS]
        _entity_cache = filtered
        _entity_cache_at = time.monotonic()
        logger.debug("ha_entities_fetched count=%d cached=%d", len(states), len(filtered))
        return filtered


def _format_entities(entities: list[dict]) -> str:
    lines = []
    for e in entities:
        entity_id = e.get("entity_id", "")
        friendly = e.get("attributes", {}).get("friendly_name", entity_id)
        state = e.get("state", "unknown")
        lines.append(f"{entity_id} ({friendly}) [{state}]")
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


async def _call_service(
    url: str, token: str, service: str, entity_id: str, service_data: dict
) -> None:
    domain, action = service.split(".", 1)
    # Guard against LLM including entity_id inside service_data, which would silently override it.
    clean_data = {k: v for k, v in service_data.items() if k != "entity_id"}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/api/services/{domain}/{action}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"entity_id": entity_id, **clean_data},
            timeout=5.0,
        )
        if response.is_error:
            logger.warning(
                "ha_service_error status=%d body=%s", response.status_code, response.text[:200]
            )
        response.raise_for_status()
