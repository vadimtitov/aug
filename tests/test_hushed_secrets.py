"""Tests for aug/utils/hushed.py — runtime secret reading.

Covers:
  - os.environ hit is returned and cached
  - hushed fallback when env var is absent
  - empty result is not cached (a later addition is picked up)
  - invalid env-var names are rejected
  - hushed subprocess failure returns empty string, not an exception
"""

from unittest.mock import patch

import pytest

from aug.utils.hushed import _cache, read_secret


@pytest.fixture(autouse=True)
def _clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def test_env_var_present_is_returned():
    """A secret in os.environ is returned immediately."""
    with patch.dict("os.environ", {"MY_SECRET": "from-env"}):
        assert read_secret("MY_SECRET") == "from-env"


def test_env_var_hit_is_cached():
    """A positive result is cached so we don't re-check on every call."""
    with patch.dict("os.environ", {"MY_SECRET": "from-env"}):
        read_secret("MY_SECRET")
    # Even after env is removed, the cached value is returned.
    assert read_secret("MY_SECRET") == "from-env"


def test_hushed_fallback_when_env_absent(monkeypatch):
    """When os.environ doesn't have it, hushed is consulted."""
    monkeypatch.delenv("LATER_SECRET", raising=False)

    def fake_read_from_hushed(name):
        assert name == "LATER_SECRET"
        return "from-hushed"

    monkeypatch.setattr("aug.utils.hushed._read_from_hushed", fake_read_from_hushed)
    assert read_secret("LATER_SECRET") == "from-hushed"


def test_empty_result_is_not_cached(monkeypatch):
    """A missing secret is retried on the next call, so a later addition is found."""
    monkeypatch.delenv("ABSENT_SECRET", raising=False)

    call_count = 0

    def fake_read_from_hushed(name):
        nonlocal call_count
        call_count += 1
        return "found" if call_count == 2 else ""

    monkeypatch.setattr("aug.utils.hushed._read_from_hushed", fake_read_from_hushed)

    assert read_secret("ABSENT_SECRET") == ""
    assert read_secret("ABSENT_SECRET") == "found"
    assert call_count == 2


def test_invalid_env_name_returns_empty():
    """Only valid env-var names are passed to hushed."""
    assert read_secret("lowercase") == ""
    assert read_secret("with-dash") == ""
    assert read_secret("") == ""


def test_hushed_subprocess_failure_returns_empty(monkeypatch):
    """A subprocess error doesn't propagate — the caller gets an empty string."""
    monkeypatch.delenv("FAILING_SECRET", raising=False)

    def failing(name):
        raise RuntimeError("boom")

    monkeypatch.setattr("aug.utils.hushed._read_from_hushed", failing)
    assert read_secret("FAILING_SECRET") == ""
