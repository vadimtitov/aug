"""Unit tests for the generate_image tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.tools.image_gen import generate_image


def _invoke(prompt: str):
    return generate_image.ainvoke(
        {"type": "tool_call", "id": "test-id", "name": "generate_image", "args": {"prompt": prompt}}
    )


def _mock_openai_response(url="https://example.com/image.png"):
    img_data = MagicMock()
    img_data.url = url
    response = MagicMock()
    response.data = [img_data]
    return response


@pytest.mark.asyncio
async def test_generate_image_success():
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_img_resp = MagicMock()
    mock_img_resp.raise_for_status = MagicMock()
    mock_img_resp.content = img_bytes
    mock_http.get = AsyncMock(return_value=mock_img_resp)

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=_mock_openai_response())

    with (
        patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai),
        patch("aug.core.tools.image_gen.httpx.AsyncClient", return_value=mock_http),
    ):
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
async def test_generate_image_no_url():
    response = MagicMock()
    response.data = []

    mock_openai = AsyncMock()
    mock_openai.images = AsyncMock()
    mock_openai.images.generate = AsyncMock(return_value=response)

    with patch("aug.core.tools.image_gen.AsyncOpenAI", return_value=mock_openai):
        result = await _invoke("empty")

    assert "url" in result.content.lower()
    assert not result.artifact.attachments
