# Skills System Design

## What is a skill

A named directory of instructions and resources, following the [agentskills.io spec](https://agentskills.io/specification).
Injected into context when the agent decides it's relevant — not always in the system prompt.

Skills replace the current `skills.md` memory file entirely.

---

## Format (agentskills.io-compatible)

```
/app/data/skills/
└── ha-automations/
    ├── SKILL.md          # required: frontmatter + instructions
    ├── scripts/          # optional: executable code
    ├── references/       # optional: detailed docs
    └── assets/           # optional: templates, resources
```

### `SKILL.md` frontmatter

| Field         | Required | Notes                                                                                                   |
|---------------|----------|---------------------------------------------------------------------------------------------------------|
| `name`        | Yes      | Lowercase, hyphens only, matches directory name                                                         |
| `description` | Yes      | What it does + when to use it. Used in index.                                                           |
| `always_on`   | No       | AUG extension stored in `metadata.always_on`. Default false. Full content injected into every system prompt. Body must be ≤ 1000 chars. |

---

## Progressive disclosure (3 tiers)

1. **Index** (~1 line per skill) — always in system prompt:
   ```
   Available skills (call get_skill to load):
   - my-skill: Description of what this skill does and when to use it
   - another-skill: Description of what this skill does and when to use it
   ```
2. **Full SKILL.md** — loaded by agent via `get_skill(name)` when relevant
3. **Sub-files** (`scripts/`, `references/`, etc.) — loaded on demand via existing file tools

`always_on: true` skills skip tier 1/2 — full content goes directly into system prompt.

---

## System prompt changes

- Remove `<skills>` section (currently reads `skills.md`)
- Add skills index section listing all non-always_on skills by name + description
- Inject full content of `always_on: true` skills
- `build_system_prompt()` in `prompts.py` reads `/app/data/skills/` directory

---

## Migration of `skills.md`

- Integration/service knowledge → individual skill files created via `save_skill`
- Personal/env facts (addresses, account details) → `user.md`
- Secret names → co-located in relevant skill files

`skills.md` file deleted. `init_memory_files()` updated accordingly.

---

## Consolidation changes

`skills.md` removed from both light and deep consolidation:

- **Light**: remove `<skills>` input/output, remove `skills.md` instruction. Personal/env facts the LLM would previously put in skills now go into `user.md` (address, household, account details).
- **Deep**: same removal. Instruction added: if operational facts appear in `user.md`, leave them there — they no longer have a dedicated file.
- Consolidation does **not** create or update skill files. Skills are explicitly authored only.

---

## Tools

### `get_skill(name: str) -> str`
Returns full content of `SKILL.md`. Error string if skill not found.

### `save_skill(name, description, body, always_on=False) -> str | None`
Creates or overwrites a skill's `SKILL.md`. Validates:
- Name format (lowercase, hyphens, no consecutive hyphens, max 64 chars)
- Description non-empty, max 1024 chars
- If `always_on=True`: body must be ≤ 1000 chars. On failure returns error string:
  `"Body is too large for always_on (N chars). Shorten to 1000 chars or pass always_on=False."`

On success returns the full content of the written `SKILL.md` so the user can verify
directly — no LLM narration of what was saved.

### `write_skill_file(skill_name, path, content) -> str | None`
Add/update any file inside a skill directory (not `SKILL.md`).
Validates path doesn't escape skill directory and doesn't target `SKILL.md`.
On success returns the full content of the written file, same rationale as `save_skill`.
On error returns an error string.

### `delete_skill(skill_name, path=None) -> str`
`path=None` → delete entire skill directory.
`path="scripts/foo.py"` → delete that file only.

---

## Implementation plan

1. **Deprecate `skills.md`** — remove from `init_memory_files()`, consolidation I/O, and system prompt builder
2. **Update consolidation prompts** — remove `<skills>` input/output tags and instructions from both light and deep prompts; add note that operational facts belong in `user.md`
3. **Skills loader** (`aug/utils/skills.py`) — `load_skills()` returning `always_on` content list and on-demand index string
4. **`build_system_prompt()`** — inject always_on skill content + on-demand index
5. **Tools** (`aug/core/tools/skills.py`) — `get_skill`, `save_skill`, `write_skill_file`, `delete_skill`
6. **Registry** — add skills tools to relevant agents
7. **Tests** — skills loader, tool validation, always_on size enforcement, path escape prevention

Steps 1–2 are removal. Steps 3–7 are additive. Each step has accompanying tests.
