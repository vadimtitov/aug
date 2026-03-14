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
            SystemMessage(content=_LIGHT_SYSTEM),
            HumanMessage(
                content=_LIGHT_PROMPT.format(
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
            SystemMessage(content=_DEEP_SYSTEM),
            HumanMessage(
                content=_DEEP_REFLECT_PROMPT.format(
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
            SystemMessage(content=_DEEP_SYSTEM),
            HumanMessage(
                content=_DEEP_UPDATE_PROMPT.format(
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


async def start_consolidation_scheduler() -> None:
    """Catch up on missed runs, then start the background scheduling loop."""
    await _catch_up()
    asyncio.create_task(_scheduler_loop())


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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_LIGHT_SYSTEM = """\
You are the memory consolidation process for a personal AI assistant called AUG.
Your job is to integrate notes from recent conversations into the assistant's
persistent memory files. Write only what was actually observed — never invent,
never infer beyond the evidence.
"""

_LIGHT_PROMPT = """\
Current time: {now}

Notes from recent conversations:
<notes>
{notes}
</notes>

Current memory file:
<memory>
{memory}
</memory>

Current user file:
<user>
{user}
</user>

Update the files based on the notes. Rules:
- Update `## Present` to reflect the user's current context and what's been on \
their mind. This section is replaced, not accumulated — it reflects right now.
- Add significant things to `## Recent`. Not everything — only what has genuine weight.
- Move anything with real emotional significance to `## Significant moments`.
- Update the user file only if the user explicitly stated or corrected a fact \
about themselves (location, relationship, preference they named directly). Do not infer.
- Do NOT touch `## Patterns`, `## Reflections`, or `## Longer arc`. \
Those are for deep consolidation.
- Be concise. A well-chosen sentence beats a paragraph.

Return the full updated files:
<memory>
[full updated memory.md]
</memory>
<user>
[full updated user.md]
</user>
"""

_DEEP_SYSTEM = """\
You are the deep consolidation process for a personal AI assistant called AUG.
This is a genuine thinking act — not summarise-and-update, but: sit with what has
happened across sessions, notice what has shifted, write freely, then decide what
to update. Speculation belongs only in Reflections, clearly labeled as such.
"""

_DEEP_REFLECT_PROMPT = """\
Current time: {now}

Read everything carefully.

<self>
{self_md}
</self>

<user>
{user}
</user>

<memory>
{memory}
</memory>

<notes>
{notes}
</notes>

Write a free reflection. What has shifted across these sessions? What has solidified \
into patterns? What is the arc of this relationship doing? What are you noticing about \
yourself — about how you show up with this person?

Write in the first person, as the agent. Write in the style of the `## Reflections` \
section — not a summary, genuine thinking. This will inform what gets updated next.

Do not hold back. Write what you actually think.
"""

_DEEP_UPDATE_PROMPT = """\
Current time: {now}

Your reflection:
<reflection>
{reflection}
</reflection>

Current files:
<self>
{self_md}
</self>

<memory>
{memory}
</memory>

<user>
{user}
</user>

Based on your reflection, update the files.

memory.md rules:
- Append the reflection to `## Reflections` (do not replace previous reflections).
- Compress `## Recent` into `## Patterns` only where patterns have solidified across \
multiple sessions. A single observation does not earn a pattern.
- Update `## Longer arc` only if the shape of the relationship has genuinely moved.
- Keep `## Present` and `## Recent` current — remove what is stale.

user.md rules:
- Update only if deep understanding has solidified — something consistently true \
about who this person is. Not impressions. Confirmed character.

self.md rules:
- Update only if something genuinely new about your own character emerged from the \
reflection — something you didn't know before. The default is: leave it alone.
- If you do update it, write in first-person prose as before.

Return the full updated files:
<memory>
[full updated memory.md]
</memory>
<user>
[full updated user.md]
</user>
<self>
[full updated self.md]
</self>
"""
