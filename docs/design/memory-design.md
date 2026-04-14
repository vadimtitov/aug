# Memory Design — AUG Personal Assistant

## Goal

Build an AI personal assistant that feels genuinely known over months and years of use.
The agent should know the user deeply, have a stable sense of its own identity, and evolve
as the relationship develops.

The mechanism: four plain-text files on disk, injected into every conversation as context,
written back by the agent periodically via background consolidation jobs and mid-conversation
notes.

---

## The files

### `self.md` — Agent identity

What it is: the agent's stable sense of self. Character, voice, values, how it relates to
this person. Written in first-person prose — not a personality description, but a living
narrative of what kind of entity this agent is and is becoming.

What goes in it:
- Core character and values
- How it handles disagreement, uncertainty, and care
- A sense of the relationship and where it's growing

What does not go here: operational rules, user facts, preferences. Those belong in `user.md`.

How it's updated: by both light consolidation (when the user gives clear behavioral
feedback — "be more concise", "less formal") and deep consolidation (when reflection
surfaces something that has genuinely shifted). The threshold is meaningful refinement,
not revolution. Small but genuine updates are welcome.

---

### `user.md` — Stable user reference

What it is: everything worth knowing about this person — stable facts, preferences, rules,
behavioural patterns, environment, tools, accounts.

What goes in it:
- Biographical facts, household, relationships
- Preferences, operational rules, invariants
- Behavioural patterns that have solidified across sessions
- Systems, tools, and accounts AUG interacts with on their behalf

What does not go here: anything with an expiry date. If it's time-sensitive or currently
active, it belongs in `context.md`.

How it's updated: nightly by light consolidation, and weekly by deep consolidation.
Each pass enforces a ~600 word budget — over-budget files must shed stale or redundant
entries to make room for new ones.

---

### `context.md` — Live working memory

What it is: what's happening right now. Active situations, upcoming events, recent notable
activity. Everything is time-anchored.

Format: all entries carry an ISO date prefix `[YYYY-MM-DD]`. Three sections:

```
## Present
[2026-04-14] Currently focused on camera stack integration.

## Upcoming
- [2026-06-08] Ye concert, GelreDome Arnhem.

## Recent
- [2026-04-12] Added Hetzner server dusya to AUG workflows.
```

Pruned aggressively on every nightly consolidation pass:
- Entries older than ~6 weeks are removed unless still clearly active
- Events whose date has passed are removed
- Entries absorbed as permanent rules in `user.md` or `self.md` are removed

Target: under 300 words. The timestamps are what enable pruning — without them the
consolidation LLM has no basis for deciding what is stale.

---

### `notes.md` — Raw capture ring buffer

What it is: timestamped raw notes taken mid-conversation by the agent via the `note` tool.
Pending consolidation.

This file is injected into the system prompt so facts noted in one session are visible to
the agent before consolidation has run. Cleared after each nightly consolidation pass.

Capped at 100 lines (ring buffer — oldest entries dropped when full).

---

### `reflections.md` — Deep consolidation diary

Written by deep consolidation only. Never injected into the runtime system prompt. A
running record of the agent's periodic reflections on the relationship and its own
identity. Append-only.

---

## Consolidation

Human memory consolidates in stages: fast tagging of salience during experience, slow
pattern extraction offline. Two consolidation passes mirror this.

### Light consolidation — nightly at 03:00 UTC

A single LLM call with full visibility across all files. Inputs:

- `notes.md` (fresh notes to process)
- `self.md`, `context.md`, `user.md` (read-only context for deduplication)

Outputs (only files that actually changed):

- `context.md` — timestamped new entries; stale entries pruned
- `user.md` — stable facts and patterns promoted from notes
- `self.md` — updated if notes contain clear behavioral feedback from the user

Cross-file deduplication is mandatory: if something is written to `self.md`, it is
removed from `user.md` and `context.md`. If written to `user.md`, it is removed from
`context.md`. The same fact never lives in two files.

The LLM is explicitly instructed to omit any file tag if that file needs no change.
Output tokens cost money; silence is the correct response when nothing has changed.

Skipped entirely if `notes.md` is empty. Notes are cleared after the pass.

### Deep consolidation — weekly on Sunday at 04:00 UTC

A two-stage process. More reflective, more philosophical. Scope: the full picture across
time.

**Stage 1 — Reflect freely**

The LLM reads all files plus `reflections.md` (the diary of past reflections) and writes
a free reflection. No updates yet — just genuine thinking. Questions asked:

- What has shifted about this person or the relationship?
- What has solidified into patterns?
- What in `self.md` feels stale, overstated, or no longer quite right?
- Has something in the relationship moved that belongs in `self.md`?

**Stage 2 — Update**

Given the reflection, the LLM decides what to update across `self.md`, `user.md`, and
`context.md`. Same cross-file deduplication rules and omit-on-no-change contract as light
consolidation. Also appends the reflection to `reflections.md`.

---

## The `note` tool

The single write mechanism during a conversation. Used liberally — the threshold is low.
If a fact, preference, correction, pattern, or operational detail might be useful to
remember next time, it gets noted.

Explicit prohibition: credentials, passwords, tokens, API keys, or anything resembling
a secret must never be recorded — even if the user mentions them directly.

Notes are raw capture. Consolidation decides where they end up, not the agent mid-session.

---

## LLM constraints and how they shape this design

**Context window is finite.**
All files must stay small enough to inject into every conversation without crowding out
the conversation itself. File budgets (300w context, 600w user) and aggressive pruning
enforce this. A concise file is more valuable than an exhaustive one.

**LLMs do not learn between sessions.**
Nothing persists unless written to disk. Every observation not written is permanently
lost. The write mechanisms are load-bearing in a way human memory is not.

**LLMs are prone to sycophancy and recency bias.**
The consolidation prompts enforce cross-session consistency: patterns must be observed
across multiple sessions, not from a single data point. `user.md` changes slowly.

**LLMs confabulate.**
The agent should write only what it has actually observed. Consolidation prompts enforce
this — speculation has no place in these files.

**LLMs have no sense of elapsed time.**
ISO date prefixes on context entries are the only temporal grounding. They enable the
pruning logic that keeps `context.md` from growing unboundedly.

---

## What this system is not

- Not a conversation log. Logs produce retrieval thinking, not relational thinking.
- Not a preference database. `user.md` captures the person, not a flat list of settings.
- Not rigidly structured. Files are free-form Markdown — the agent writes like a person
  with a notebook, not like a system filling in fields.
- Not static. All files are living documents the agent reads and writes as a first-class
  part of its operation.
