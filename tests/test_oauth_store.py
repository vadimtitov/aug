"""Tests for aug/core/oauth/store.py — encrypted token persistence.

Behaviors under test:
  - encrypt/decrypt round trip
  - ciphertext bound to (provider, account): a row moved between accounts fails
  - a wrong-sized key is rejected loudly rather than silently weakening encryption
"""

import pytest
from cryptography.exceptions import InvalidTag

from aug.core.oauth.store import decrypt, encrypt


def test_round_trip():
    blob = encrypt("rt-secret", "spotify", "primary")

    assert blob != b"rt-secret"
    assert decrypt(blob, "spotify", "primary") == "rt-secret"


def test_ciphertext_is_bound_to_provider_and_account():
    """A row copied onto another (provider, account) must fail, not silently work.

    Without AAD binding, moving a ciphertext between rows would authenticate the
    agent as the wrong account against the wrong API.
    """
    blob = encrypt("rt-secret", "spotify", "primary")

    with pytest.raises(InvalidTag):
        decrypt(blob, "strava", "primary")

    with pytest.raises(InvalidTag):
        decrypt(blob, "spotify", "work")


def test_same_plaintext_encrypts_differently_each_time():
    """A fresh nonce per call — otherwise equal tokens are visibly equal in the DB."""
    assert encrypt("rt-secret", "spotify", "primary") != encrypt("rt-secret", "spotify", "primary")


def test_wrong_sized_key_is_rejected(monkeypatch):
    monkeypatch.setenv("OAUTH_ENCRYPTION_KEY", "c2hvcnQ=")  # 5 bytes

    with pytest.raises(RuntimeError, match="32 bytes"):
        encrypt("rt-secret", "spotify", "primary")
