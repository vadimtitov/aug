"""Tests for the BASE_URL guard in aug/config.py.

BASE_URL is what every OAuth link and redirect URI is built from, and it defaults
to empty. Left unchecked, a misconfigured deployment mints links with no host and
nobody finds out until a user taps one.

Behaviors under test:
  - production requires an https BASE_URL
  - development is unaffected
"""

import pytest

from aug.config import Settings

_REQUIRED = {
    "API_KEY": "k",
    "LLM_API_KEY": "k",
    "LLM_BASE_URL": "http://litellm:4000",
    "DATABASE_URL": "postgresql+asyncpg://u:p@db:5432/d",
}


def _settings(**overrides) -> Settings:
    return Settings(**(_REQUIRED | overrides))


def test_production_rejects_a_plain_http_base_url():
    with pytest.raises(ValueError, match="BASE_URL"):
        _settings(DEBUG=False, BASE_URL="http://aug.example.com")


def test_production_rejects_an_empty_base_url():
    """The default. Silently mints hostless links, which is the worst failure mode."""
    with pytest.raises(ValueError, match="BASE_URL"):
        _settings(DEBUG=False, BASE_URL="")


def test_production_accepts_https():
    assert _settings(DEBUG=False, BASE_URL="https://aug.example.com").base_url == (
        "https://aug.example.com"
    )


def test_development_is_unaffected():
    """Local dev runs on plain http, and often with no BASE_URL at all."""
    assert _settings(DEBUG=True, BASE_URL="http://localhost:8012").base_url == (
        "http://localhost:8012"
    )
    assert _settings(DEBUG=True, BASE_URL="").base_url == ""
