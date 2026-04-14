"""Application settings loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False  # True → human-readable logs; does NOT affect auth
    # Set to True in local dev to skip Telegram initData HMAC verification.
    # Must be False in production — any other value is a critical security risk.
    DEV_AUTH_BYPASS: bool = False

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

    # Gmail OAuth (optional — tool is disabled if absent)
    GMAIL_CLIENT_ID: str | None = None
    GMAIL_CLIENT_SECRET: str | None = None

    # Portainer (optional — portainer tools disabled if absent)
    PORTAINER_URL: str | None = None
    PORTAINER_API_TOKEN: str | None = None

    # Base URL used for OAuth redirect URIs and auth links sent to users.
    # Defaults to auto-detected LAN IP on port 8012.
    BASE_URL: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_url(self) -> str:
        return self.BASE_URL.rstrip("/")

    # Browser tool — CDP URL of the remote Chromium instance
    BROWSER_CDP_URL: str | None = None

    # Home Assistant (optional — HA reflex disabled if absent)
    # Accepts HASS_URL, HA_URL, or HOMEASSISTANT_URL; HASS_TOKEN or HOMEASSISTANT_TOKEN
    HASS_URL: str | None = None
    HA_URL: str | None = None
    HOMEASSISTANT_URL: str | None = None
    HASS_TOKEN: str | None = None
    HOMEASSISTANT_TOKEN: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ha_url(self) -> str | None:
        return (self.HASS_URL or self.HA_URL or self.HOMEASSISTANT_URL or "").rstrip("/") or None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ha_token(self) -> str | None:
        return self.HASS_TOKEN or self.HOMEASSISTANT_TOKEN or None

    @model_validator(mode="after")
    def _validate_paired_settings(self) -> "Settings":
        if bool(self.GMAIL_CLIENT_ID) != bool(self.GMAIL_CLIENT_SECRET):
            raise ValueError(
                "GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must both be set or both absent"
            )
        if bool(self.PORTAINER_URL) != bool(self.PORTAINER_API_TOKEN):
            raise ValueError(
                "PORTAINER_URL and PORTAINER_API_TOKEN must both be set or both absent"
            )
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded lazily and cached."""
    return Settings()  # type: ignore[call-arg]
