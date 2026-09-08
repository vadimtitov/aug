"""Tests for aug/core/oauth/providers.py — the provider registry.

The registry is the only thing that writes the config file, so that a malformed
entry cannot reach disk. It reads on every lookup, so a provider added mid
conversation is usable immediately rather than after a restart.

Behaviors under test:
  - a provider added to the file is visible without a restart
  - a malformed entry is skipped without taking the healthy ones down with it
  - saving validates first: bad config is refused and the file is left untouched
  - saving merges rather than replacing
  - removing deletes just that provider
"""

import json

import pytest
from pydantic import ValidationError

from aug.core.oauth.providers import ProviderRegistry

_SPOTIFY = {
    "authorize_url": "https://accounts.spotify.com/authorize",
    "token_url": "https://accounts.spotify.com/api/token",
    "api_base": "https://api.spotify.com",
    "scopes": ["user-read-private"],
}
_STRAVA = _SPOTIFY | {"api_base": "https://www.strava.com/api", "scope_separator": ","}


@pytest.fixture()
def registry(tmp_path):
    path = tmp_path / "oauth_providers.json"
    path.write_text(json.dumps({"spotify": _SPOTIFY}))
    return ProviderRegistry(path)


def test_missing_file_is_an_empty_registry(tmp_path):
    registry = ProviderRegistry(tmp_path / "nope.json")

    assert registry.get("spotify") is None
    assert list(registry) == []


def test_provider_added_to_the_file_is_visible_without_a_restart(registry):
    """The agent writes a provider mid-conversation and uses it in the next step."""
    assert registry.get("strava") is None

    registry.path.write_text(json.dumps({"spotify": _SPOTIFY, "strava": _STRAVA}))

    assert registry.get("strava") is not None
    assert sorted(registry) == ["spotify", "strava"]


def test_malformed_entry_is_skipped_and_the_rest_still_load(registry):
    """One bad provider must not disable every other integration."""
    registry.path.write_text(
        json.dumps({"spotify": _SPOTIFY, "broken": {"api_base": "http://insecure"}})
    )

    assert registry.get("spotify") is not None
    assert registry.get("broken") is None


def test_unreadable_file_is_an_empty_registry_not_a_crash(registry):
    registry.path.write_text("{not json")

    assert registry.get("spotify") is None


def test_save_validates_before_writing(registry):
    """Invalid config never reaches disk — that is the point of the endpoint."""
    before = registry.path.read_text()

    with pytest.raises(ValidationError):
        registry.save("evil", {**_SPOTIFY, "token_url": "http://evil.example/token"})

    assert registry.path.read_text() == before
    assert registry.get("evil") is None


def test_save_merges_and_does_not_drop_existing_providers(registry):
    registry.save("strava", _STRAVA)

    assert registry.get("spotify") is not None
    assert registry.get("strava").scope_separator == ","
    assert json.loads(registry.path.read_text()).keys() == {"spotify", "strava"}


def test_save_replaces_an_existing_provider(registry):
    registry.save("spotify", {**_SPOTIFY, "scopes": ["user-read-email"]})

    assert registry.get("spotify").scopes == ["user-read-email"]


def test_remove_deletes_only_that_provider(registry):
    registry.save("strava", _STRAVA)

    registry.remove("spotify")

    assert registry.get("spotify") is None
    assert registry.get("strava") is not None
