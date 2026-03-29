"""Tests for chat agent preprocess and image expansion."""

import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aug.core.agents.chat_agent import AugAgent, TimeAwareChatAgent, _expand_images
from aug.core.state import AgentState

_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}}
_TEXT_BLOCK = {"type": "text", "text": "What do you see?"}
_MULTIMODAL_CONTENT = [_IMAGE_BLOCK, _TEXT_BLOCK]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aug_agent():
    with (
        patch("aug.core.llm.get_settings") as mock_settings,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value="system prompt"),
    ):
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        yield AugAgent(model="gpt-4o")


@pytest.fixture
def time_aware_agent():
    with patch("aug.core.llm.get_settings") as mock_settings:
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        yield TimeAwareChatAgent(model="gpt-4o", system_prompt="You are helpful.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(content: str | list) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=content)],
        thread_id="test",
        interface="telegram",
    )


# ---------------------------------------------------------------------------
# preprocess: _stamp preserves multimodal content
# ---------------------------------------------------------------------------


def test_aug_agent_preprocess_preserves_image_blocks(aug_agent):
    update = aug_agent.preprocess(_state(_MULTIMODAL_CONTENT))

    last = update.messages[-1]
    assert isinstance(last.content, list), "multimodal content was flattened to string"
    image_blocks = [b for b in last.content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_time_aware_agent_preprocess_preserves_image_blocks(time_aware_agent):
    update = time_aware_agent.preprocess(_state(_MULTIMODAL_CONTENT))

    last = update.messages[-1]
    assert isinstance(last.content, list), "multimodal content was flattened to string"
    image_blocks = [b for b in last.content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_aug_agent_preprocess_stamps_plain_text(aug_agent):
    update = aug_agent.preprocess(_state("hello"))
    last = update.messages[-1]
    assert isinstance(last.content, str)
    assert "hello" in last.content
    assert "UTC" in last.content  # timestamp injected


# ---------------------------------------------------------------------------
# _expand_images: re-inlines [[img:path|mime]] markers at respond time
# ---------------------------------------------------------------------------


def test_expand_images_inlines_marker_in_last_human_message(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"imgdata")
    marker = f"[[img:{img_path}|image/jpeg]]"

    messages = [HumanMessage(content=marker)]
    result = _expand_images(messages)

    last = result[-1]
    assert isinstance(last.content, list)
    image_blocks = [b for b in last.content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"imgdata"


def test_expand_images_preserves_caption_text(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")
    content = f"[[img:{img_path}|image/jpeg]]\n\nWhat do you see?"

    result = _expand_images([HumanMessage(content=content)])

    last = result[-1]
    text_blocks = [b for b in last.content if b.get("type") == "text"]
    assert any("What do you see?" in b["text"] for b in text_blocks)


def test_expand_images_does_not_expand_historical_messages(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")
    marker = f"[[img:{img_path}|image/jpeg]]"

    messages = [
        HumanMessage(content=marker),  # historical
        AIMessage(content="I see a photo."),
        HumanMessage(content="follow up"),  # current (no marker)
    ]
    result = _expand_images(messages)

    # Historical message must remain unchanged
    assert result[0].content == marker
    # Current message has no marker, no change
    assert result[2].content == "follow up"


def test_expand_images_file_not_found_falls_back_to_text(tmp_path):
    missing = str(tmp_path / "gone.jpg")
    marker = f"[[img:{missing}|image/jpeg]]"

    result = _expand_images([HumanMessage(content=marker)])

    last = result[-1]
    assert isinstance(last.content, list)
    assert not any(b.get("type") == "image_url" for b in last.content)
    assert any("not found" in b.get("text", "").lower() for b in last.content)


def test_expand_images_no_marker_returns_unchanged():
    messages = [HumanMessage(content="plain text")]
    result = _expand_images(messages)
    assert result is messages  # same object, no copy made
