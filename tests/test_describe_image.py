"""Tests for the describe_image tool factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from aug.core.tools.describe_image import make_describe_image_tool


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"imgdata")
    return path


@pytest.fixture
def describe_tool():
    with patch("aug.core.tools.describe_image.build_chat_model") as mock_build:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="A red car on a street."))
        mock_build.return_value = mock_llm
        yield make_describe_image_tool("gemini-2.0-flash"), mock_build, mock_llm


@pytest.mark.asyncio
async def test_describe_image_returns_description(image_file, describe_tool):
    tool, _, _ = describe_tool
    result = await tool.ainvoke({"path": str(image_file), "question": "What colour is the car?"})
    assert "red car" in result.lower()


@pytest.mark.asyncio
async def test_describe_image_file_not_found(describe_tool, tmp_path):
    tool, _, _ = describe_tool
    result = await tool.ainvoke(
        {"path": str(tmp_path / "missing.jpg"), "question": "What is this?"}
    )
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_describe_image_vision_model_failure(image_file, describe_tool):
    tool, _, mock_llm = describe_tool
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("model error"))
    result = await tool.ainvoke({"path": str(image_file), "question": "What is this?"})
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_describe_image_uses_specified_model(image_file):
    with patch("aug.core.tools.describe_image.build_chat_model") as mock_build:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="A cat."))
        mock_build.return_value = mock_llm
        tool = make_describe_image_tool("gemini-2.0-flash")
        await tool.ainvoke({"path": str(image_file), "question": "What is this?"})
        mock_build.assert_called_once_with("gemini-2.0-flash")
