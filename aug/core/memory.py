"""Memory file management and consolidation (the dream).

Files, in workspace/memory/ (the dream writes nothing outside it):
  MEMORY.md      — The index: one line per file.  A snapshot goes into each session's
                   prompt; the agent reads the files themselves when it needs them.
  self.md        — AUG's own sense of itself, as it evolves.  The persona the agent runs
                   with is AGENTS.md, which the dream never touches.
  user.md        — Who the user is: profile, preferences, behavioural rules, env facts.
  context.md     — Volatile: Present / Upcoming / Recent with dated entries.
  reflections.md — Deep consolidation diary.
  notes.md       — Ring buffer of raw notes. Cleared after each consolidation.

Consolidation, run as jobs (aug/core/jobs.py) through a harness's oneshot():
  Light (nightly)  — folds notes into context.md, user.md, self.md.
  Deep  (weekly)   — reflects across all files; updates self.md, user.md, context.md,
                     appends to reflections.md.
Both may rewrite MEMORY.md.
"""

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from aug.core.harness.base import Harness
from aug.core.harness.types import ModelTier
from aug.core.prompts import (
    CONSOLIDATION_DEEP_REFLECT_PROMPT,
    CONSOLIDATION_DEEP_SYSTEM,
    CONSOLIDATION_DEEP_UPDATE_PROMPT,
    CONSOLIDATION_LIGHT_PROMPT,
    CONSOLIDATION_LIGHT_SYSTEM,
)
from aug.utils.data import MEMORY_DIR, MEMORY_INDEX

logger = logging.getLogger(__name__)

_NOTES_MAX_LINES = 100
# MEMORY.md is cut to this many characters, so a long index can't bloat every prompt.
MEMORY_INDEX_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_memory_files() -> None:
    """Create memory files with defaults if they don't exist yet."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _init("self.md", "")
    _init("user.md", _USER_MD)
    _init("context.md", _CONTEXT_MD)
    _init("reflections.md", "")
    _init("notes.md", "")


def append_note(content: str) -> None:
    """Append a timestamped note, dropping oldest entries beyond the cap."""
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    path = MEMORY_DIR / "notes.md"
    current = path.read_text().strip() if path.exists() else ""
    lines = [line for line in current.splitlines() if line.strip()]
    lines.append(f"[{now}] {content}")
    if len(lines) > _NOTES_MAX_LINES:
        lines = lines[-_NOTES_MAX_LINES:]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


async def run_light_consolidation(harness: Harness) -> bool:
    """Nightly pass: fold notes into context.md, user.md, and optionally self.md.

    Passes all memory files as context so the LLM can make cross-file deduplication
    decisions in a single call. Returns True if it ran, False if no notes to process.
    """
    notes = _read("notes.md")
    if not notes.strip():
        logger.info("Light consolidation: no notes, skipping.")
        return False

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    text = await harness.oneshot(
        CONSOLIDATION_LIGHT_PROMPT.format(
            notes=notes,
            self_md=_read("self.md"),
            context=_read("context.md"),
            user=_read("user.md"),
            memory_index=_read(MEMORY_INDEX.name),
            now=now,
        ),
        ModelTier.FAST,
        system=CONSOLIDATION_LIGHT_SYSTEM,
    )
    if updated := _extract("context", text):
        _write("context.md", updated)
    if updated := _extract("user", text):
        _write("user.md", updated)
    if updated := _extract("self", text):
        _write("self.md", updated)
    if updated := _extract("memory_index", text):
        _write(MEMORY_INDEX.name, cap_memory_index(updated))

    _drop_notes(notes)
    logger.info("Light consolidation complete.")
    return True


async def run_deep_consolidation(harness: Harness) -> None:
    """Weekly pass: reflect across all files, update what has genuinely shifted."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    notes = _read("notes.md")

    # Stage 1 — reflect freely, no updates yet.
    reflection = await harness.oneshot(
        CONSOLIDATION_DEEP_REFLECT_PROMPT.format(
            self_md=_read("self.md"),
            user=_read("user.md"),
            context=_read("context.md"),
            reflections=_read("reflections.md"),
            notes=notes,
            now=now,
        ),
        ModelTier.FAST,
        system=CONSOLIDATION_DEEP_SYSTEM,
    )

    # Stage 2 — given the reflection, decide what to update.
    text = await harness.oneshot(
        CONSOLIDATION_DEEP_UPDATE_PROMPT.format(
            reflection=reflection,
            self_md=_read("self.md"),
            user=_read("user.md"),
            context=_read("context.md"),
            memory_index=_read(MEMORY_INDEX.name),
            now=now,
        ),
        ModelTier.FAST,
        system=CONSOLIDATION_DEEP_SYSTEM,
    )
    if updated := _extract("user", text):
        _write("user.md", updated)
    if updated := _extract("self", text):
        _write("self.md", updated)
    if updated := _extract("context", text):
        _write("context.md", updated)
    if updated := _extract("memory_index", text):
        _write(MEMORY_INDEX.name, cap_memory_index(updated))
    if new_reflection := _extract("new_reflection", text):
        existing = _read("reflections.md")
        _write("reflections.md", (existing + "\n\n" + new_reflection).strip())

    _drop_notes(notes)
    logger.info("Deep consolidation complete.")


def cap_memory_index(text: str) -> str:
    """*text* cut to MEMORY_INDEX_MAX_CHARS, at a line break."""
    text = text.strip()
    if len(text) <= MEMORY_INDEX_MAX_CHARS:
        return text
    cut = text[:MEMORY_INDEX_MAX_CHARS]
    return cut[: cut.rfind("\n")] if "\n" in cut else cut


# ---------------------------------------------------------------------------
# File defaults
# ---------------------------------------------------------------------------

_USER_MD = """\
Nothing is known about this person yet.
"""

_CONTEXT_MD = """\
## Present
[YYYY-MM-DD] Brief description of current focus.

## Upcoming
- [YYYY-MM-DD] Upcoming event or deadline.

## Recent
- [YYYY-MM-DD] Recent notable event or completed situation.
"""

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read(name: str) -> str:
    path: Path = MEMORY_DIR / name
    return path.read_text().strip() if path.exists() else ""


def _write(name: str, content: str) -> None:
    """Write through a temporary file, so a session starting meanwhile never reads half."""
    partial = MEMORY_DIR / f".{name}.partial"
    partial.write_text(content.strip() + "\n")
    os.replace(partial, MEMORY_DIR / name)


def _drop_notes(consumed: str) -> None:
    """Remove the notes a run read, keeping any written while it waited on the model."""
    done = set(consumed.splitlines())
    kept = [line for line in _read("notes.md").splitlines() if line not in done]
    _write("notes.md", "\n".join(kept))


def _extract(tag: str, text: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _init(name: str, default: str) -> None:
    path: Path = MEMORY_DIR / name
    if not path.exists():
        path.write_text(default)
