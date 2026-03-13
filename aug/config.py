"""Application settings loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic import computed_field
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
    # Comma-separated chat IDs allowed to use the bot. If empty, all chats are allowed.
    TELEGRAM_ALLOWED_CHAT_IDS: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_chat_ids(self) -> set[int]:
        if not self.TELEGRAM_ALLOWED_CHAT_IDS:
            return set()
        try:
            return {int(s.strip()) for s in self.TELEGRAM_ALLOWED_CHAT_IDS.split(",") if s.strip()}
        except ValueError as e:
            raise ValueError(
                f"TELEGRAM_ALLOWED_CHAT_IDS must be comma-separated integers: {e}"
            ) from e

    # Brave Search (optional — tool is disabled if key is absent)
    BRAVE_API_KEY: str | None = None

    # Browser tool — CDP URL of the remote Chromium instance
    BROWSER_CDP_URL: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded lazily and cached."""
    return Settings()  # type: ignore[call-arg]
