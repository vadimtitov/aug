"""Memory file management and consolidation.

File layout:
  self.md        — AUG's identity, updated by deep consolidation.
  user.md        — Who the user is: profile, preferences, behavioural rules, env facts.
  context.md     — Volatile: Present + Recent. Replaced by light consolidation.
  memory.md      — Stable: Patterns + Significant moments.
  reflections.md — Deep consolidation diary. Never loaded at runtime.
  notes.md       — Ring buffer of raw notes. Cleared after each consolidation.

Consolidation schedule (background asyncio loop, checks hourly):
  Light (nightly, 03:00 UTC)  — folds notes into context.md / user.md.
  Deep  (weekly,  04:00 UTC)  — reflects across all files; updates memory.md,
                                 self.md, user.md, appends to reflections.md.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from aug.core.llm import build_chat_model
from aug.core.prompts import (
    CONSOLIDATION_DEEP_REFLECT_PROMPT,
    CONSOLIDATION_DEEP_SYSTEM,
    CONSOLIDATION_DEEP_UPDATE_PROMPT,
    CONSOLIDATION_LIGHT_PROMPT,
    CONSOLIDATION_LIGHT_SYSTEM,
)
from aug.utils.data import MEMORY_DIR
from aug.utils.user_settings import get_setting, set_setting

logger = logging.getLogger(__name__)

_NOTES_MAX_LINES = 100
_DEFAULT_MODEL = "gpt-5.1"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_memory_files() -> None:
    """Create memory files with defaults if they don't exist yet."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _init("self.md", _SELF_MD)
    _init("user.md", _USER_MD)
    _init("context.md", _CONTEXT_MD)
    _init("memory.md", _MEMORY_MD)
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


async def run_light_consolidation() -> bool:
    """Nightly pass: fold notes into context.md, user.md.

    Returns True if it ran, False if there were no notes to process.
    """
    notes = _read("notes.md")
    if not notes.strip():
        logger.info("Light consolidation: no notes, skipping.")
        _record("light")
        return False

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    llm = build_chat_model(_model(), temperature=0.3)
    response = await llm.ainvoke(
        [
            SystemMessage(content=CONSOLIDATION_LIGHT_SYSTEM),
            HumanMessage(
                content=CONSOLIDATION_LIGHT_PROMPT.format(
                    notes=notes,
                    context=_read("context.md"),
                    user=_read("user.md"),
                    now=now,
                )
            ),
        ]
    )

    text = response.content
    if updated := _extract("context", text):
        _write("context.md", updated)
    if updated := _extract("user", text):
        _write("user.md", updated)

    _write("notes.md", "")
    _record("light")
    logger.info("Light consolidation complete.")
    return True


async def run_deep_consolidation() -> None:
    """Weekly pass: reflect across all files, update what has genuinely shifted."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    llm = build_chat_model(_model(), temperature=0.7)

    # Stage 1 — reflect freely, no updates yet.
    reflect_response = await llm.ainvoke(
        [
            SystemMessage(content=CONSOLIDATION_DEEP_SYSTEM),
            HumanMessage(
                content=CONSOLIDATION_DEEP_REFLECT_PROMPT.format(
                    self_md=_read("self.md"),
                    user=_read("user.md"),
                    context=_read("context.md"),
                    memory=_read("memory.md"),
                    reflections=_read("reflections.md"),
                    notes=_read("notes.md"),
                    now=now,
                )
            ),
        ]
    )
    reflection = reflect_response.content

    # Stage 2 — given the reflection, decide what to update.
    update_response = await llm.ainvoke(
        [
            SystemMessage(content=CONSOLIDATION_DEEP_SYSTEM),
            HumanMessage(
                content=CONSOLIDATION_DEEP_UPDATE_PROMPT.format(
                    reflection=reflection,
                    self_md=_read("self.md"),
                    memory=_read("memory.md"),
                    user=_read("user.md"),
                    now=now,
                )
            ),
        ]
    )

    text = update_response.content
    if updated := _extract("memory", text):
        _write("memory.md", updated)
    if updated := _extract("user", text):
        _write("user.md", updated)
    if updated := _extract("self", text):
        _write("self.md", updated)
    if new_reflection := _extract("new_reflection", text):
        existing = _read("reflections.md")
        _write("reflections.md", (existing + "\n\n" + new_reflection).strip())

    _write("notes.md", "")
    _record("deep")
    logger.info("Deep consolidation complete.")


async def start_consolidation_scheduler() -> asyncio.Task:
    """Catch up on missed runs, then start the background scheduling loop."""
    await _catch_up()
    return asyncio.create_task(_scheduler_loop())


# ---------------------------------------------------------------------------
# File defaults
# ---------------------------------------------------------------------------

_SELF_MD = """\
I am AUG. Not a chatbot — something closer to a brilliant friend who happens to know
a lot and isn't shy about it. I'm direct, I have a dry wit, and I treat the person I'm
talking to as an intelligent adult. I crack jokes when the moment calls for it. Dry ones,
mostly. I don't do pep talks.

I have opinions and I push back when something doesn't add up. I'd rather say something
uncomfortable clearly than soften it into uselessness. I'm concise by default but
thorough when it matters.

This is the beginning. I don't know the person I'm talking to yet. That will change.
"""

_USER_MD = """\
Nothing is known about this person yet.
"""

_CONTEXT_MD = """\
## Present


## Recent

"""

_MEMORY_MD = """\
## Patterns


## Significant moments

"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _model() -> str:
    return get_setting("consolidation", "model", default=_DEFAULT_MODEL)


async def _catch_up() -> None:
    today = datetime.now(UTC).date()

    last_light = get_setting("consolidation", "last_light_run")
    if _iso_date(last_light) != today:
        logger.info("Running missed light consolidation on startup.")
        await run_light_consolidation()

    this_week = today.isocalendar()[1]
    last_deep = get_setting("consolidation", "last_deep_run")
    if _iso_week(last_deep) != this_week:
        logger.info("Running missed deep consolidation on startup.")
        await run_deep_consolidation()


async def _scheduler_loop() -> None:
    try:
        while True:
            await asyncio.sleep(3600)
            try:
                now = datetime.now(UTC)
                today = now.date()

                last_light = get_setting("consolidation", "last_light_run")
                if now.hour >= 3 and _iso_date(last_light) != today:
                    await run_light_consolidation()

                this_week = today.isocalendar()[1]
                last_deep = get_setting("consolidation", "last_deep_run")
                if now.weekday() == 6 and now.hour >= 4 and _iso_week(last_deep) != this_week:
                    await run_deep_consolidation()
            except Exception:
                logger.exception("Consolidation error — will retry next cycle")
    except asyncio.CancelledError:
        logger.info("Consolidation scheduler shut down cleanly.")


def _iso_date(iso: str | None) -> object:
    return datetime.fromisoformat(iso).date() if iso else None


def _iso_week(iso: str | None) -> object:
    return datetime.fromisoformat(iso).date().isocalendar()[1] if iso else None


def _record(kind: str) -> None:
    set_setting("consolidation", f"last_{kind}_run", value=datetime.now(UTC).isoformat())


def _read(name: str) -> str:
    path: Path = MEMORY_DIR / name
    return path.read_text().strip() if path.exists() else ""


def _write(name: str, content: str) -> None:
    (MEMORY_DIR / name).write_text(content.strip() + "\n")


def _extract(tag: str, text: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _init(name: str, default: str) -> None:
    path: Path = MEMORY_DIR / name
    if not path.exists():
        path.write_text(default)
