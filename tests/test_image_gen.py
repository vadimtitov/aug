"""Unit tests for the generate_image tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.tools.image_gen import generate_image


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
async def test_generate_image_success():
    response, img_bytes = _mock_openai_response_b64()

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("a red cat")

    assert "a red cat" in result.content
    assert result.artifact.attachments
    assert result.artifact.attachments[0].data == img_bytes


@pytest.mark.asyncio
async def test_generate_image_openai_error():
    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(side_effect=Exception("model not found"))

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("a blue dog")

    assert "failed" in result.content.lower()
    assert not result.artifact.attachments


@pytest.mark.asyncio
async def test_generate_image_no_data():
    response = MagicMock()
    response.data = []

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("empty")

    assert "did not return" in result.content.lower()
    assert not result.artifact.attachments
