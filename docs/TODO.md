# TODO

## Context compaction

Long conversation threads accumulate context that balloons the prompt. Need a strategy to
compact/summarise older messages without losing continuity. Options: sliding window,
summarisation node in the graph, or a dedicated compaction step triggered at token threshold.

---

## Agent interruption

Ability to interrupt the agent mid-run:
- **Soft interrupt** — inject additional information while the agent is working (e.g. "actually
  ignore X"). When would the agent see it? Likely only at the next tool-call boundary or after
  the current LLM turn completes — need to think through the mechanics.
- **Hard stop** — cancel the current run entirely. `/stop` command in Telegram is the obvious
  UX. Requires hooking into LangGraph's cancellation or the underlying async task.

---

## Skills

Something similar to Claude Code skills or OpenClaw skills. Community-generated skills are
compelling (OpenClaw has a large library). Two angles:
1. **User-defined / community skills** — load and execute named skill definitions.
2. **Self-generated skills** — as part of the nightly consolidation process, the agent
   identifies recurring task patterns and writes new skills for itself.

---

## Memory system overhaul

Current memory dumps too much into the prompt; a lot of it is duplicated. Needs review:
- Share the current system prompt with Claude Code and ask for suggested solutions.
- Possible directions: retrieval-based memory (embed + similarity search instead of full
  dump), tiered memory (hot/warm/cold), deduplication pass during consolidation.

---

## Settings system redesign

Replace the current `settings.json` / `get_setting` / `set_setting` implementation
with a properly typed, Pydantic-based settings model.

**Motivation:**
The current implementation reads from disk on every `get_setting()` call. There is no
schema, no validation, and no guarantee that what is in memory reflects what is on disk.
Nested path access via variadic string tuples is fragile and untyped.

**Goals:**
- Define the full settings schema as a Pydantic model — typed, validated, with defaults.
- Load once from disk into a singleton in memory; all reads go through that object.
- Writes are synchronised back to disk (write-through), keeping disk and memory consistent.
- No read-from-disk on every access.

**Open questions to resolve before implementing:**
- How to handle concurrent writes safely (async lock? single writer pattern?).
- How to handle schema evolution — what happens when new fields are added and the
  existing `settings.json` on disk predates them (default injection on load?).
- Whether settings should be hot-reloadable without a restart, or restart-to-reload
  is acceptable.
- Whether per-user / per-chat settings (currently nested under `"telegram.chats.<id>"`)
  fit naturally into a typed model or need a separate dynamic store.
