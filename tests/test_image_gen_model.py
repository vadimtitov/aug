"""Tests for IMAGE_GEN_MODEL migration to settings.json."""

from unittest.mock import patch

from aug.core.tools.image_gen import _get_image_gen_model

_DEFAULT = "gpt-image-1.5"


def test_returns_default_when_not_configured() -> None:
    with patch("aug.core.tools.image_gen.get_setting", return_value=None):
        assert _get_image_gen_model() == _DEFAULT


def test_returns_configured_model() -> None:
    with patch("aug.core.tools.image_gen.get_setting", return_value="dall-e-3"):
        assert _get_image_gen_model() == "dall-e-3"
