"""Tests for aug/utils/hushed.py — the hushed CLI client."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aug.utils.hushed import list_secret_names


@pytest.mark.asyncio
async def test_list_secret_names_parses_names():
    result = MagicMock(returncode=0, stdout="GITHUB_TOKEN\nDATABASE_URL\n", stderr="")
    with patch("aug.utils.hushed.subprocess.run", return_value=result):
        names = await list_secret_names()
    assert names == {"GITHUB_TOKEN", "DATABASE_URL"}


@pytest.mark.asyncio
async def test_list_secret_names_returns_empty_on_failure():
    with patch(
        "aug.utils.hushed.subprocess.run", side_effect=subprocess.TimeoutExpired("hushed", 10)
    ):
        assert await list_secret_names() == set()


@pytest.mark.asyncio
async def test_list_secret_names_returns_empty_on_nonzero_exit():
    result = MagicMock(returncode=1, stdout="", stderr="hushed: not found")
    with patch("aug.utils.hushed.subprocess.run", return_value=result):
        assert await list_secret_names() == set()
