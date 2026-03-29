"""Tests for chat agent preprocess — especially multimodal message handling."""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from aug.core.agents.chat_agent import AugAgent, TimeAwareChatAgent
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
# Tests
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
