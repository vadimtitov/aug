"""Memory consolidation — light (nightly) and deep (weekly) passes.

Two functions:
  run_light_consolidation() — processes notes into memory.md and user.md.
  run_deep_consolidation()  — full reflection pass across all three files.

Both are plain async functions that call the LLM directly via build_chat_model().
No LangGraph, no tools, no streaming — just a structured LLM call.

Scheduling: a background asyncio loop checks every hour. On startup, missed
runs are caught up using last-run timestamps stored in settings.json.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage

from aug.core.llm import build_chat_model
from aug.core.memory import MEMORY_DIR
from aug.core.prompts import (
    CONSOLIDATION_DEEP_REFLECT_PROMPT,
    CONSOLIDATION_DEEP_SYSTEM,
    CONSOLIDATION_DEEP_UPDATE_PROMPT,
    CONSOLIDATION_LIGHT_PROMPT,
    CONSOLIDATION_LIGHT_SYSTEM,
)
from aug.utils.user_settings import get_setting, set_setting

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.1"


def _model() -> str:
    return get_setting("consolidation", "model", default=_DEFAULT_MODEL)


async def run_light_consolidation() -> None:
    """Nightly pass: fold notes into memory.md and user.md."""
    notes = _read("notes.md")
    if not notes.strip():
        logger.info("Light consolidation: no notes, skipping.")
        _record("light")
        return

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    llm = build_chat_model(_model(), temperature=0.3)
    response = await llm.ainvoke(
        [
            SystemMessage(content=CONSOLIDATION_LIGHT_SYSTEM),
            HumanMessage(
                content=CONSOLIDATION_LIGHT_PROMPT.format(
                    notes=notes,
                    memory=_read("memory.md"),
                    user=_read("user.md"),
                    now=now,
                )
            ),
        ]
    )

    text = response.content
    if updated := _extract("memory", text):
        _write("memory.md", updated)
    if updated := _extract("user", text):
        _write("user.md", updated)

    _write("notes.md", "")
    _record("light")
    logger.info("Light consolidation complete.")


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
                    memory=_read("memory.md"),
                    user=_read("user.md"),
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

    _write("notes.md", "")
    _record("deep")
    logger.info("Deep consolidation complete.")


async def start_consolidation_scheduler() -> asyncio.Task:
    """Catch up on missed runs, then start the background scheduling loop."""
    await _catch_up()
    return asyncio.create_task(_scheduler_loop())


async def _catch_up() -> None:
    today = datetime.now(UTC).date()

    last_light = get_setting("consolidation", "last_light_run")
    if _iso_date(last_light) != today:
        logger.info("Running missed light consolidation on startup.")
        await run_light_consolidation()

    this_week = today.isocalendar()[1]
    last_deep = get_setting("consolidation", "last_deep_run")
    last_deep_week = _iso_week(last_deep)
    if last_deep_week != this_week:
        logger.info("Running missed deep consolidation on startup.")
        await run_deep_consolidation()


async def _scheduler_loop() -> None:
    try:
        while True:
            await asyncio.sleep(3600)
            now = datetime.now(UTC)
            today = now.date()

            last_light = get_setting("consolidation", "last_light_run")
            if now.hour >= 3 and _iso_date(last_light) != today:
                await run_light_consolidation()

            this_week = today.isocalendar()[1]
            last_deep = get_setting("consolidation", "last_deep_run")
            if now.weekday() == 6 and now.hour >= 4 and _iso_week(last_deep) != this_week:
                await run_deep_consolidation()
    except asyncio.CancelledError:
        logger.info("Consolidation scheduler shut down cleanly.")


def _iso_date(iso: str | None):
    return datetime.fromisoformat(iso).date() if iso else None


def _iso_week(iso: str | None):
    return datetime.fromisoformat(iso).date().isocalendar()[1] if iso else None


def _record(kind: str) -> None:
    set_setting("consolidation", f"last_{kind}_run", value=datetime.now(UTC).isoformat())


def _read(name: str) -> str:
    path = MEMORY_DIR / name
    return path.read_text().strip() if path.exists() else ""


def _write(name: str, content: str) -> None:
    (MEMORY_DIR / name).write_text(content.strip() + "\n")


def _extract(tag: str, text: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
