"""Tests for vision_description_model integration in AugAgent."""

from unittest.mock import patch

import pytest

from aug.core.agents.chat_agent import AugAgent


@pytest.fixture
def agent_with_vision_model():
    with (
        patch("aug.core.llm.get_settings") as mock_settings,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value="prompt"),
    ):
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        yield AugAgent(model="glm-5", vision_description_model="gemini-2.0-flash")


@pytest.fixture
def agent_without_vision_model():
    with (
        patch("aug.core.llm.get_settings") as mock_settings,
        patch("aug.core.agents.chat_agent.build_system_prompt", return_value="prompt"),
    ):
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        yield AugAgent(model="gpt-4o")


def test_describe_image_tool_injected_when_vision_model_set(agent_with_vision_model):
    tool_names = [t.name for t in agent_with_vision_model.tools]
    assert "describe_image" in tool_names


def test_describe_image_tool_not_injected_by_default(agent_without_vision_model):
    tool_names = [t.name for t in agent_without_vision_model.tools]
    assert "describe_image" not in tool_names
