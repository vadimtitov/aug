"""API key authentication dependency.

Usage — apply at router level so every route in the router is protected:

    router = APIRouter(dependencies=[Depends(require_api_key)])

The ``/health`` and ``/telegram/webhook`` routes are excluded by design
(they are registered *before* the protected routers).
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from aug.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency — raises 401 if the API key is missing or wrong."""
    if not api_key or api_key != get_settings().API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
