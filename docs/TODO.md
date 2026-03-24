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

## Browser agent context gap

The browser agent operates in isolation — it only sees what the main agent explicitly
passes in its task prompt. The main agent has to predict upfront what context the browser
will need (credentials, addresses, preferences, operational rules), which isn't always
possible. This creates a structural gap.

Potential directions:
- **Richer initial context injection** — automatically prepend relevant sections from
  `skills.md` (e.g. Deliveroo, Amazon) and `user.md` to every browser task, so the
  browser agent has standard operational context without the main agent having to think
  about it.
- **Browser-to-main interruption** — allow the browser agent to pause its own run and
  surface a question or blocker back to the main agent (e.g. "I need the OTP code" or
  "Which address should I use?"). Requires a callback or structured yield mechanism in
  the browser tool interface.
- **Dynamic context requests** — the browser agent detects that it is missing something
  (e.g. hits a login form with no credentials in scope) and requests specific context from
  the main agent rather than failing or guessing.
- **Shared tool access** — give the browser agent access to a read-only subset of AUG's
  tools (e.g. a `lookup_secret` or `get_user_preference` tool) so it can self-resolve
  gaps without interrupting the main agent.

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
