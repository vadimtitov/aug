"""Unit tests for consolidation utilities."""

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.core.memory import (
    _extract,
    _iso_date,
    _iso_week,
    _read,
    _write,
    run_deep_consolidation,
    run_light_consolidation,
)

# ---------------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------------


def test_extract_found() -> None:
    text = "<memory>some content here</memory>"
    assert _extract("memory", text) == "some content here"


def test_extract_multiline() -> None:
    text = "<user>\nline one\nline two\n</user>"
    assert _extract("user", text) == "line one\nline two"


def test_extract_missing_tag() -> None:
    assert _extract("memory", "no tags here") is None


def test_extract_trims_whitespace() -> None:
    assert _extract("self", "<self>  trimmed  </self>") == "trimmed"


# ---------------------------------------------------------------------------
# _iso_date / _iso_week
# ---------------------------------------------------------------------------


def test_iso_date_parses() -> None:
    assert _iso_date("2024-03-15T03:00:00+00:00") == date(2024, 3, 15)


def test_iso_date_none() -> None:
    assert _iso_date(None) is None


def test_iso_week_parses() -> None:
    # 2024-03-15 is week 11
    assert _iso_week("2024-03-15T00:00:00+00:00") == 11


def test_iso_week_none() -> None:
    assert _iso_week(None) is None


# ---------------------------------------------------------------------------
# _read / _write
# ---------------------------------------------------------------------------


def test_read_returns_empty_for_missing_file(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        assert _read("nonexistent.md") == ""


def test_read_returns_stripped_content(tmp_path: Path) -> None:
    (tmp_path / "user.md").write_text("  hello  \n\n")
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        assert _read("user.md") == "hello"


def test_write_creates_file(tmp_path: Path) -> None:
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        _write("user.md", "new content")
    assert (tmp_path / "user.md").read_text() == "new content\n"


def test_write_overwrites(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("old")
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        _write("notes.md", "")
    assert (tmp_path / "notes.md").read_text() == "\n"


# ---------------------------------------------------------------------------
# run_light_consolidation
# ---------------------------------------------------------------------------


def _mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return llm


@pytest.mark.asyncio
async def test_light_consolidation_skips_when_no_notes(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("")
    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model") as mock_build,
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()
        mock_build.assert_not_called()


@pytest.mark.asyncio
async def test_light_consolidation_writes_context_and_user(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] user likes cats")
    (tmp_path / "context.md").write_text("## Present\n\n## Recent\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<context>## Present\nfocused on cats\n## Recent\n</context><user>Likes cats.</user>"

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=_mock_llm(response)),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    assert "focused on cats" in (tmp_path / "context.md").read_text()
    assert "Likes cats." in (tmp_path / "user.md").read_text()
    assert (tmp_path / "notes.md").read_text().strip() == ""


@pytest.mark.asyncio
async def test_light_consolidation_writes_env_facts_to_user(tmp_path: Path) -> None:
    """Operational/env facts (previously skills.md) now go into user.md."""
    (tmp_path / "notes.md").write_text("[2026-01-01] you have Home Assistant at HA_URL")
    (tmp_path / "context.md").write_text("## Present\n\n## Recent\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<user>Home Assistant: HA_URL + HASS_TOKEN. Query entity IDs before use.</user>"

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=_mock_llm(response)),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    assert "Home Assistant" in (tmp_path / "user.md").read_text()


@pytest.mark.asyncio
async def test_light_consolidation_skips_missing_tags(tmp_path: Path) -> None:
    """If model omits a tag, that file is left unchanged."""
    (tmp_path / "notes.md").write_text("[2026-01-01] something minor")
    (tmp_path / "context.md").write_text("original context")
    (tmp_path / "user.md").write_text("original user")

    response = "<context>updated context</context>"  # user omitted

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=_mock_llm(response)),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    assert "updated context" in (tmp_path / "context.md").read_text()
    assert (tmp_path / "user.md").read_text() == "original user"


# ---------------------------------------------------------------------------
# Light consolidation — full redesign behaviors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_light_consolidation_writes_self_md_when_tag_present(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] be more concise")
    (tmp_path / "self.md").write_text("I am AUG.")
    (tmp_path / "context.md").write_text("## Present\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<self>I am AUG. More concise now.</self>"

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=_mock_llm(response)),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    assert "More concise now." in (tmp_path / "self.md").read_text()


@pytest.mark.asyncio
async def test_light_consolidation_does_not_write_self_md_when_tag_absent(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] user likes cats")
    (tmp_path / "self.md").write_text("original self")
    (tmp_path / "context.md").write_text("## Present\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<user>Likes cats.</user>"  # no <self> tag

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=_mock_llm(response)),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    assert (tmp_path / "self.md").read_text().strip() == "original self"


@pytest.mark.asyncio
async def test_light_consolidation_passes_self_md_to_prompt(tmp_path: Path) -> None:
    """Light consolidation passes self.md as read-only context for deduplication."""
    (tmp_path / "notes.md").write_text("[2026-01-01] something")
    (tmp_path / "self.md").write_text("I am AUG. Dry wit.")
    (tmp_path / "context.md").write_text("## Present\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    captured_prompt: list[str] = []

    def capture_llm(response_text: str) -> MagicMock:
        llm = MagicMock()

        async def ainvoke(messages: list) -> MagicMock:
            for m in messages:
                if hasattr(m, "content"):
                    captured_prompt.append(m.content)
            return MagicMock(content=response_text)

        llm.ainvoke = ainvoke
        return llm

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=capture_llm("")),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_light_consolidation()

    full_prompt = "\n".join(captured_prompt)
    assert "Dry wit." in full_prompt


def test_light_consolidation_prompt_contains_dedup_instruction() -> None:
    from aug.core.prompts import CONSOLIDATION_LIGHT_PROMPT

    prompt_lower = CONSOLIDATION_LIGHT_PROMPT.lower()
    assert "dedup" in prompt_lower or "duplicate" in prompt_lower


def test_light_consolidation_prompt_contains_omit_on_no_change_instruction() -> None:
    from aug.core.prompts import CONSOLIDATION_LIGHT_PROMPT

    assert "omit" in CONSOLIDATION_LIGHT_PROMPT.lower()


def test_light_consolidation_prompt_contains_file_size_budget() -> None:
    from aug.core.prompts import CONSOLIDATION_LIGHT_PROMPT

    assert (
        "word" in CONSOLIDATION_LIGHT_PROMPT.lower()
        or "budget" in CONSOLIDATION_LIGHT_PROMPT.lower()
    )


# ---------------------------------------------------------------------------
# Light consolidation prompt — timestamps and pruning
# ---------------------------------------------------------------------------


def test_light_consolidation_prompt_instructs_timestamp_format() -> None:
    from aug.core.prompts import CONSOLIDATION_LIGHT_PROMPT

    assert (
        "yyyy-mm-dd" in CONSOLIDATION_LIGHT_PROMPT.lower() or "[2026-" in CONSOLIDATION_LIGHT_PROMPT
    )


def test_light_consolidation_prompt_instructs_pruning() -> None:
    from aug.core.prompts import CONSOLIDATION_LIGHT_PROMPT

    assert (
        "prune" in CONSOLIDATION_LIGHT_PROMPT.lower()
        or "remove" in CONSOLIDATION_LIGHT_PROMPT.lower()
    )


# ---------------------------------------------------------------------------
# Deep consolidation — identity evolution prompts
# ---------------------------------------------------------------------------


def test_deep_reflect_prompt_asks_about_stale_self() -> None:
    from aug.core.prompts import CONSOLIDATION_DEEP_REFLECT_PROMPT

    prompt_lower = CONSOLIDATION_DEEP_REFLECT_PROMPT.lower()
    assert "stale" in prompt_lower or "no longer" in prompt_lower


def test_deep_reflect_prompt_receives_context() -> None:
    from aug.core.prompts import CONSOLIDATION_DEEP_REFLECT_PROMPT

    assert "{context}" in CONSOLIDATION_DEEP_REFLECT_PROMPT


def test_deep_update_prompt_lowered_self_threshold() -> None:
    from aug.core.prompts import CONSOLIDATION_DEEP_UPDATE_PROMPT

    prompt_lower = CONSOLIDATION_DEEP_UPDATE_PROMPT.lower()
    # Should mention refinement, not just major shifts
    assert "refin" in prompt_lower or "partially" in prompt_lower


@pytest.mark.asyncio
async def test_deep_consolidation_writes_self_md_when_tag_present(tmp_path: Path) -> None:
    (tmp_path / "self.md").write_text("I am AUG.")
    (tmp_path / "user.md").write_text("Name: V")
    (tmp_path / "context.md").write_text("## Present\nworking")
    (tmp_path / "reflections.md").write_text("")
    (tmp_path / "notes.md").write_text("")

    reflect_response = "Some reflection."
    update_response = "<self>Updated identity.</self><new_reflection>ref</new_reflection>"

    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content=reflect_response),
            MagicMock(content=update_response),
        ]
    )

    with (
        patch("aug.core.memory.MEMORY_DIR", tmp_path),
        patch("aug.core.memory.build_chat_model", return_value=llm),
        patch("aug.core.memory.load_settings"),
        patch("aug.core.memory.save_state"),
    ):
        await run_deep_consolidation()

    assert "Updated identity." in (tmp_path / "self.md").read_text()


# ---------------------------------------------------------------------------
# run_deep_consolidation
# ---------------------------------------------------------------------------
