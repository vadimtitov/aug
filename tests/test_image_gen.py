"""Unit tests for the generate_image tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.tools import blocked_paths
from aug.core.tools.image_gen import edit_image, generate_image


def _invoke(prompt: str):
    return generate_image.ainvoke(
        {"type": "tool_call", "id": "test-id", "name": "generate_image", "args": {"prompt": prompt}}
    )


def _mock_openai_response_b64(data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100):
    img_data = MagicMock()
    img_data.b64_json = base64.b64encode(data).decode()
    img_data.url = None
    response = MagicMock()
    response.data = [img_data]
    return response, data


@pytest.mark.asyncio
async def test_generate_image_success(sent_files):
    response, img_bytes = _mock_openai_response_b64()

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("a red cat")

    assert "a red cat" in result.content
    assert len(sent_files) == 1
    assert sent_files[0].media_type == "image/png"
    assert sent_files[0].path.read_bytes() == img_bytes
    assert str(sent_files[0].path) in result.content  # so the agent can edit it later


@pytest.mark.asyncio
async def test_generate_image_openai_error(sent_files):
    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(side_effect=Exception("model not found"))

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("a blue dog")

    assert "failed" in result.content.lower()
    assert not sent_files


@pytest.mark.asyncio
async def test_generate_image_no_data(sent_files):
    response = MagicMock()
    response.data = []

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("empty")

    assert "did not return" in result.content.lower()
    assert not sent_files


@pytest.mark.asyncio
async def test_edit_image_refuses_a_blocked_path(tmp_path, monkeypatch):
    key = tmp_path / "keys" / "id_ed25519"
    key.parent.mkdir()
    key.write_text("PRIVATE KEY")
    monkeypatch.setattr(blocked_paths, "BLOCKED_PATHS", (tmp_path / "keys",))

    with patch("aug.core.tools.image_gen.AsyncOpenAI") as openai:
        result = await edit_image.ainvoke({"source_path": str(key), "prompt": "make it blue"})

    assert "NOT edited" in result
    openai.assert_not_called()


@pytest.mark.asyncio
async def test_generate_image_outside_a_chat_says_nothing_was_sent():
    response, _ = _mock_openai_response_b64()
    mock_openai = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("a red cat")

    assert "did NOT send" in result.content
