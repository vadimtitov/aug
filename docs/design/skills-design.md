# Skills

Skills are named sets of instructions the agent loads on demand. They follow the [agentskills.io spec](https://agentskills.io/specification).

---

## Structure

```
workspace/skills/
└── my-skill/
    ├── SKILL.md          # required: frontmatter + instructions
    ├── scripts/          # optional: executable scripts
    ├── references/       # optional: detailed docs
    └── assets/           # optional: templates, resources
```

### `SKILL.md` frontmatter

```yaml
---
name: my-skill
description: What this skill does and when to use it.
---

Instructions go here.
```

| Field | Required | Notes |
|-------|----------|-------|
| `name` | Yes | Lowercase, hyphens only, matches directory name, max 64 chars |
| `description` | Yes | Shown in the index so the agent knows when to load it. Max 1024 chars. |

Any other agentskills.io field (`license`, `compatibility`, `metadata`, `allowed-tools`) is
allowed and kept when the skill is edited. A skill must still work on a harness that
ignores its extra fields.

---

## How the agent uses skills

1. Every session's prompt includes a one-line index of all skills — the agent sees each skill's name and description. The index is built when the session starts, so a skill added or edited mid-chat appears from the next session.
2. When a skill is relevant, the agent calls `get_skill(name)`, which returns the skill's folder (absolute path, so bundled scripts can be run) and its full instructions.
3. Skills describe what to do ("run a shell command", "search the web"), not which AUG tool does it, so they work on every harness. Anything that must always be in the prompt belongs in `AGENTS.md`, not in a skill.

---

## Managing skills

The agent can create, update, and delete skills via tools. You can also ask it directly:

> "Create a skill for managing my Home Assistant automations"
> "Update the shopping skill to also handle returns"
> "Show me all skills" → use `/skills` in Telegram

Skills persist on disk under `workspace/skills/` and survive container restarts.
