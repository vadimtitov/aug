"""Settings router — GET/PUT /settings, GET /models, GET /settings/knowledge-base/fixed."""

import logging
from dataclasses import asdict

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from aug.api.security import require_api_key
from aug.config import get_settings
from aug.core.kb.service import KB_FIXED_MODELS
from aug.utils.file_settings import AppSettings, load_settings, save_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["settings"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/settings")
async def read_settings() -> dict:
    """Return the full user settings object."""
    return load_settings().model_dump()


@router.put("/settings")
async def write_settings(data: dict) -> dict:
    """Overwrite the full user settings object."""
    save_settings(AppSettings.model_validate(data))
    return data


@router.get("/settings/knowledge-base/fixed")
async def knowledge_base_fixed_models() -> dict[str, str]:
    """The KB models that cannot be picked, for read-only display.  Kept out of
    /settings so they are never written to settings.json."""
    return asdict(KB_FIXED_MODELS)


@router.get("/models")
async def list_models() -> list[str]:
    """Return available model IDs from the LiteLLM proxy."""
    settings = get_settings()
    base_url = settings.LLM_BASE_URL.rstrip("/")
    try:
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            body = response.json()
            return [item["id"] for item in body.get("data", [])]
    except httpx.HTTPError as exc:
        logger.warning("models_proxy_failed error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LiteLLM proxy unavailable or returned an error.",
        ) from exc
