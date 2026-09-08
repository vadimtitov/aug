"""OAuth provider registry — declarative config for any OAuth 2.0 provider.

Providers are described in ``/app/data/oauth_providers.json`` rather than in code,
so adding one is a config entry plus two secrets in hushed.  See
``docs/design/oauth-design.md``.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from aug.utils.data import DATA_DIR

logger = logging.getLogger(__name__)

PROVIDERS_FILE = DATA_DIR / "oauth_providers.json"


class ProviderConfig(BaseModel):
    """One OAuth 2.0 provider.  All URLs must be HTTPS."""

    authorize_url: str
    token_url: str
    api_base: str
    scopes: list[str]
    revoke_url: str | None = None
    scope_separator: str = " "
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)
    token_auth_method: str = "body"
    issuer: str | None = None
    client_id_env: str | None = None
    client_secret_env: str | None = None

    @field_validator("authorize_url", "token_url", "api_base", "revoke_url")
    @classmethod
    def _https_only(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError(f"must be https:// — got {value!r}")
        return value


class ProviderRegistry:
    """The provider config file, read on every lookup and written only through here.

    Reading each time means a provider added mid-conversation is usable in the next
    step instead of after a restart; the file is small and page-cached, so there is
    nothing to gain by holding a copy and an invalidation rule.

    Writing only through ``save`` is what keeps the file well-formed: config is
    validated before it can reach disk, and the agent gets the field-level error
    back instead of discovering a silently skipped entry three steps later.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, name: str) -> ProviderConfig | None:
        """The named provider, or None if it is absent or invalid."""
        return self._load().get(name)

    def save(self, name: str, config: dict) -> ProviderConfig:
        """Validate and store one provider, leaving the others untouched.

        Raises ``ValidationError`` without writing anything if the config is bad.
        """
        validated = ProviderConfig(**config)
        entries = self._raw()
        entries[name] = validated.model_dump(exclude_none=True)
        self._write(entries)
        logger.info("oauth provider %r saved", name)
        return validated

    def remove(self, name: str) -> bool:
        """Forget a provider.  False if it was not there."""
        entries = self._raw()
        if entries.pop(name, None) is None:
            return False
        self._write(entries)
        logger.info("oauth provider %r removed", name)
        return True

    def __iter__(self):
        return iter(self._load())

    def __contains__(self, name: object) -> bool:
        return name in self._load()

    def _raw(self) -> dict:
        """The file's contents, unvalidated — the basis for a merge."""
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.exception("oauth providers file unreadable — treating as empty")
            return {}

    def _load(self) -> dict[str, ProviderConfig]:
        """Every valid provider.  A malformed entry is skipped, not fatal.

        The file can still be hand-edited, so this must tolerate anything on disk —
        one bad provider must not disable every other integration.
        """
        providers: dict[str, ProviderConfig] = {}
        for name, entry in self._raw().items():
            try:
                providers[name] = ProviderConfig(**entry)
            except Exception as exc:
                logger.error("oauth provider %r skipped — invalid config: %s", name, exc)
        return providers

    def _write(self, entries: dict) -> None:
        """Replace the file atomically, so a crash cannot leave it truncated."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(self.path)
