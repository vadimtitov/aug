# Skills

Skills are named sets of instructions the agent loads on demand. They follow the [agentskills.io spec](https://agentskills.io/specification).

---

## Structure

```
data/skills/
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
| `metadata.always_on` | No | If `"true"`, full content is injected into every system prompt. Body must be ≤ 1000 chars. |

---

## How the agent uses skills

1. Every conversation includes a one-line index of all skills — the agent sees each skill's name and description.
2. When a skill is relevant, the agent calls `get_skill(name)` to load the full instructions.
3. `always_on` skills skip this — their full content is always in the system prompt.

---

## Managing skills

The agent can create, update, and delete skills via tools. You can also ask it directly:

> "Create a skill for managing my Home Assistant automations"
> "Update the shopping skill to also handle returns"
> "Show me all skills" → use `/skills` in Telegram

Skills persist on disk under `data/skills/` and survive container restarts.
