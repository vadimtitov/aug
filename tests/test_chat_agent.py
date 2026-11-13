"""Tests for chat agent preprocess and image expansion."""

import base64
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

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
        context_id="test",
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


@pytest.mark.asyncio
async def test_expand_images_inlines_marker_in_last_human_message(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"imgdata")
    marker = f"[[img:{img_path}|image/jpeg]]"

    messages = [HumanMessage(content=marker)]
    result = await _expand_images(messages, vision=True)

    last = result[-1]
    assert isinstance(last.content, list)
    image_blocks = [b for b in last.content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"imgdata"


@pytest.mark.asyncio
async def test_expand_images_preserves_caption_text(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")
    content = f"[[img:{img_path}|image/jpeg]]\n\nWhat do you see?"

    result = await _expand_images([HumanMessage(content=content)], vision=True)

    last = result[-1]
    text_blocks = [b for b in last.content if b.get("type") == "text"]
    assert any("What do you see?" in b["text"] for b in text_blocks)


@pytest.mark.asyncio
async def test_expand_images_expands_every_message_the_same_way_each_call(tmp_path):
    """History keeps its images: expanding all markers identically on every call keeps
    the prompt prefix byte-stable, so the cache holds across turns."""
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")
    marker = f"[[img:{img_path}|image/jpeg]]"
    first_turn = [HumanMessage(content=marker, id="h1"), AIMessage(content="I see a photo.")]
    second_turn = [*first_turn, HumanMessage(content="follow up", id="h2")]

    before = await _expand_images(first_turn, vision=True)
    after = await _expand_images(second_turn, vision=True)

    assert before[0].content == after[0].content
    assert any(b.get("type") == "image_url" for b in after[0].content)
    assert after[2].content == "follow up"
    assert first_turn[0].content == marker  # the stored history is untouched


@pytest.mark.asyncio
async def test_expand_images_hints_the_path_next_to_the_image(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")

    result = await _expand_images(
        [HumanMessage(content=f"[[img:{img_path}|image/jpeg]]")], vision=True
    )

    texts = [b["text"] for b in result[0].content if b.get("type") == "text"]
    assert f"[Image from user: {img_path}]" in texts


@pytest.mark.asyncio
async def test_expand_images_without_vision_gives_only_the_hint(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    Path(img_path).write_bytes(b"x")

    result = await _expand_images(
        [HumanMessage(content=f"[[img:{img_path}|image/jpeg]]")], vision=False
    )

    assert result[0].content == [{"type": "text", "text": f"[Image from user: {img_path}]"}]


@pytest.mark.asyncio
async def test_expand_images_file_not_found_falls_back_to_text(tmp_path):
    missing = str(tmp_path / "gone.jpg")
    marker = f"[[img:{missing}|image/jpeg]]"

    result = await _expand_images([HumanMessage(content=marker)], vision=True)

    last = result[-1]
    assert isinstance(last.content, list)
    assert not any(b.get("type") == "image_url" for b in last.content)
    assert any("not found" in b.get("text", "").lower() for b in last.content)


@pytest.mark.asyncio
async def test_expand_images_no_marker_returns_unchanged():
    messages = [HumanMessage(content="plain text")]
    result = await _expand_images(messages, vision=True)
    assert result == messages


# ---------------------------------------------------------------------------
# with_extra_tools
# ---------------------------------------------------------------------------


@tool
def _base_tool() -> str:
    """A base tool for tests."""
    return "base"


@tool
def _extra_tool() -> str:
    """An extra tool for tests."""
    return "extra"


def test_with_extra_tools_keeps_settings_and_adds_tools():
    with patch("aug.core.llm.get_settings") as mock_settings:
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        base = AugAgent(
            model="gpt-4o",
            tools=[_base_tool],
            temperature=0.0,
            recursion_limit=50,
            vision_description_model="gemini-2.5-flash",
            compaction_model="gpt-4o-mini",
            context_window=500_000,
        )
        extended = base.with_extra_tools([_extra_tool])

    assert extended._settings == base._settings
    assert extended.recursion_limit == 50
    # describe_image is re-added from the settings, not copied, so it appears once.
    assert [t.name for t in extended.tools] == ["_base_tool", "_extra_tool", "describe_image"]
    assert [t.name for t in base.tools] == ["_base_tool", "describe_image"]


def test_with_extra_tools_copies_every_constructor_setting():
    """A new AugAgent parameter must be added to _settings too, or copies drop it."""
    with patch("aug.core.llm.get_settings") as mock_settings:
        mock_settings.return_value.LLM_API_KEY = "test-key"
        mock_settings.return_value.LLM_BASE_URL = "http://localhost:4000"
        agent = AugAgent(model="gpt-4o")
    params = set(inspect.signature(AugAgent.__init__).parameters) - {"self", "tools"}
    assert set(agent._settings) == params
