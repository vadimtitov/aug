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

1. ✅ **Deprecate `skills.md`** — remove from `init_memory_files()`, consolidation I/O, and system prompt builder
2. ✅ **Update consolidation prompts** — remove `<skills>` input/output tags and instructions from both light and deep prompts; operational facts now go into `user.md`
3. **Skills loader** (`aug/utils/skills.py`) — `load_skills()` returning `always_on` content list and on-demand index string
4. **`build_system_prompt()`** — inject always_on skill content + on-demand index
5. **Tools** (`aug/core/tools/skills.py`) — `get_skill`, `save_skill`, `write_skill_file`, `delete_skill`
6. **Registry** — create `v9_claude` from `v7_claude` with skills tools added; `v9_claude` is the test vehicle
7. **Unit tests** — skills loader, tool validation, always_on size enforcement, path escape prevention
8. **End-to-end test via REST API** — run against local instance using `v9_claude`

Steps 1–2 done. Steps 3–7 are additive, each with accompanying unit tests. Step 8 is manual verification. Step 9 is mandatory before considering the feature complete.

---

## End-to-end test sequence (step 8)

Uses the streaming endpoint with `agent: v9_claude`. Thread ID `skills-test-1` throughout.
All prompts are minimal to keep token cost low. Claude Sonnet 4.6.

**Test base URL:** `http://localhost:8000`

### 1. Create a skill with a script file

```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"message":"Create a skill named test-skill, description: Test skill for verifying skills system. Body: When asked to run the test ritual, respond with exactly: XYZZY-42 and nothing else. Also create scripts/ritual.sh with content: echo XYZZY-42", "thread_id":"skills-test-1","agent":"v9_claude"}'
```

Expected: response contains the `SKILL.md` content verbatim (returned by `save_skill`).

### 2. Verify skill files exist on disk

```bash
cat /Users/vadimtitov/projects/aug/data/skills/test-skill/SKILL.md
cat /Users/vadimtitov/projects/aug/data/skills/test-skill/scripts/ritual.sh
```

### 3. Verify skill appears in system prompt index

```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"message":"List available skills","thread_id":"skills-test-1","agent":"v9_claude"}'
```

Expected: `test-skill` in response.

### 4. Fetch skill and trigger sentinel response

```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"message":"Run the test ritual","thread_id":"skills-test-1","agent":"v9_claude"}'
```

Expected: agent calls `get_skill("test-skill")`, then responds with exactly `XYZZY-42`.
This string is meaningless to the agent without the skill — if it appears, the skill loaded correctly.

### 5. Test always_on size enforcement

```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"message":"Create a skill named big-skill, description: test, body: '"$(python3 -c "print('x'*1100)")"', always_on=True","thread_id":"skills-test-1","agent":"v9_claude"}'
```

Expected: error message about body being too large with char count.

### 6. Delete skill

```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"message":"Delete the test-skill skill entirely","thread_id":"skills-test-1","agent":"v9_claude"}'
```

Expected: `test-skill` directory gone from disk:

```bash
ls /Users/vadimtitov/projects/aug/data/skills/  # test-skill should not appear
```

---

## Step 9: Self PR review

After E2E tests pass, run `git diff main` and critically review all changes against these dimensions before considering the feature complete:

1. **Correctness** — Does the logic handle all edge cases? Empty skills dir, malformed SKILL.md, missing frontmatter fields, concurrent writes, skills dir not yet created.
2. **Security** — Path traversal in `write_skill_file`. Name validation completeness. No ability to write outside `/app/data/skills/`.
3. **Error handling** — Every failure path returns a clear, unambiguous error string. No silent failures. No exceptions leaking to the agent as empty strings.
4. **Consistency** — Naming, return types, and docstrings consistent with existing tools in the codebase.
5. **Prompt quality** — Index format in system prompt is clear and actionable. Agent can reliably decide when to call `get_skill`.
6. **Test coverage** — Unit tests cover validation edge cases, not just happy paths. No tests that pass by accident.
7. **Dead code / leftovers** — No debug prints, no commented-out code, no unused imports, no stale references to `skills.md`.

Address any findings before shipping.
