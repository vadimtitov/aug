"""Application settings loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Auth
    API_KEY: str

    # LLM — all requests routed through LiteLLM proxy
    LLM_API_KEY: str
    LLM_BASE_URL: str

    # Database
    DATABASE_URL: str  # postgresql+asyncpg://user:password@host:5432/dbname

    # Telegram (all optional — bot is disabled if token is absent)
    TELEGRAM_BOT_TOKEN: str | None = None

    # Brave Search (optional — tool is disabled if key is absent)
    BRAVE_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded lazily and cached."""
    return Settings()  # type: ignore[call-arg]
