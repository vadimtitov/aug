"""Unit tests for consolidation utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from aug.core.harness.types import ModelTier
from aug.core.memory import (
    _extract,
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


class _Harness:
    """Records every oneshot() call and answers from *replies* in turn."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, ModelTier, str]] = []

    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG, *, system=""):
        self.calls.append((prompt, tier, system))
        return self.replies.pop(0) if self.replies else ""


@pytest.mark.asyncio
async def test_light_consolidation_skips_when_no_notes(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("")
    harness = _Harness()
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        assert await run_light_consolidation(harness) is False
    assert harness.calls == []


@pytest.mark.asyncio
async def test_light_consolidation_writes_context_and_user(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] user likes cats")
    (tmp_path / "context.md").write_text("## Present\n\n## Recent\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<context>## Present\nfocused on cats\n## Recent\n</context><user>Likes cats.</user>"

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_Harness(response))

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

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_Harness(response))

    assert "Home Assistant" in (tmp_path / "user.md").read_text()


@pytest.mark.asyncio
async def test_light_consolidation_skips_missing_tags(tmp_path: Path) -> None:
    """If model omits a tag, that file is left unchanged."""
    (tmp_path / "notes.md").write_text("[2026-01-01] something minor")
    (tmp_path / "context.md").write_text("original context")
    (tmp_path / "user.md").write_text("original user")

    response = "<context>updated context</context>"  # user omitted

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_Harness(response))

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

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_Harness(response))

    assert "More concise now." in (tmp_path / "self.md").read_text()


@pytest.mark.asyncio
async def test_light_consolidation_does_not_write_self_md_when_tag_absent(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] user likes cats")
    (tmp_path / "self.md").write_text("original self")
    (tmp_path / "context.md").write_text("## Present\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    response = "<user>Likes cats.</user>"  # no <self> tag

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_Harness(response))

    assert (tmp_path / "self.md").read_text().strip() == "original self"


@pytest.mark.asyncio
async def test_light_consolidation_passes_self_md_to_prompt(tmp_path: Path) -> None:
    """Light consolidation passes self.md as read-only context for deduplication."""
    (tmp_path / "notes.md").write_text("[2026-01-01] something")
    (tmp_path / "self.md").write_text("I am AUG. Dry wit.")
    (tmp_path / "context.md").write_text("## Present\n")
    (tmp_path / "user.md").write_text("Nothing known.")

    harness = _Harness("")
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(harness)

    ((prompt, tier, system),) = harness.calls
    assert tier is ModelTier.FAST
    assert system
    assert "Dry wit." in prompt


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

    harness = _Harness(reflect_response, update_response)
    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_deep_consolidation(harness)

    assert "Updated identity." in (tmp_path / "self.md").read_text()
    assert "Some reflection." in harness.calls[1][0]  # the update sees the reflection
    assert [tier for _, tier, _ in harness.calls] == [ModelTier.FAST, ModelTier.FAST]


# ---------------------------------------------------------------------------
# run_deep_consolidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_note_written_while_the_dream_runs_is_kept(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("[2026-01-01] old note\n")

    class _NoteMidRun(_Harness):
        async def oneshot(self, prompt, tier=ModelTier.STRONG, *, system=""):
            with (tmp_path / "notes.md").open("a") as f:
                f.write("[2026-01-02] new note\n")
            return "<user>u</user>"

    with patch("aug.core.memory.MEMORY_DIR", tmp_path):
        await run_light_consolidation(_NoteMidRun())

    assert (tmp_path / "notes.md").read_text() == "[2026-01-02] new note\n"
