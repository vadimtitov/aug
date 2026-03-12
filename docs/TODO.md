# TODO

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
