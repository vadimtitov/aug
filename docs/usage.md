# Using AUG

Talk to the bot like a person. Send text, voice notes, photos, files or your location, live or once. Write while it's working and your message joins the turn.

It replies with text and files, and messages you first when a task fires or background work finishes.

## Commands

| | |
|---|---|
| `/clear` | Start fresh |
| `/stop` | Stop the current turn |
| `/status` | Model, context, plan limits |
| `/compact` | Summarise to free context |
| `/version` | Switch this chat's agent |
| `/claude_login`, `/codex_login` | Sign in (private chat) |
| `/secret` | Store a key; your message is deleted |
| `/skills` | Browse skills |
| `/ssh`, `/approvals` | SSH targets and saved approvals |
| `/consolidate` | Run the nightly memory pass now |
| `/prompt` | The session's system prompt, as a file |

Want parallel conversations? Make a private group with topics and add the bot. Each topic is its own chat. [Setup](setup-guides/telegram-forum-setup.md)

## Memory

Everything it is and knows lives in plain files in `data/workspace/`, and you can edit them.

| | |
|---|---|
| `AGENTS.md` | Persona and approach. Yours to rewrite |
| `memory/user.md` | Who you are |
| `memory/context.md` | What's going on now |
| `memory/self.md` | Its sense of itself |
| `memory/notes.md` | Notes from today, waiting for the night |

Every night it folds the day's notes into the files; once a week it reviews them all. Each conversation starts with an index of the files, and the agent opens what it needs. Edits apply from the next `/clear`. [Design](design/memory-design.md)

## Tasks

"Remind me at 9 tomorrow to call the bank." "Every Monday at 8, summarise my unread email." One-off or recurring. When a task fires, it sends the text or runs the agent with it, in the chat it came from.

Other systems can push in with `POST /hooks/push`. [Design](design/webhook-design.md)

## Skills

Ask for one: "make a skill for filing my receipts." Skills are folders with a `SKILL.md` ([agentskills.io](https://agentskills.io/specification)) under `data/workspace/skills/`, and load when a task needs them. Python scripts can declare their dependencies inline. [Design](design/skills-design.md)

## Tools

On top of what Claude Code and Codex bring:

| | |
|---|---|
| **Web** | Reads any page; a stealth browser takes over when a site blocks bots |
| **Files** | What you send lands in the workspace; it sends files back |
| **SSH** | Your servers. AUG makes and installs the key; every command needs your OK |
| **Gmail** | Several accounts. [Setup](setup-guides/gmail-setup.md) |
| **Images** | Generate and edit (OpenAI key) |
| **Portainer** | Containers and stacks. [Setup](setup-guides/portainer-setup.md) |
| **MCP** | Finds, installs and removes MCP servers |

The container has Python with pandas and matplotlib, Node, ffmpeg, pandoc, ImageMagick, Tesseract, jq and the GitHub CLI.

## REST API

Your own apps and scripts can talk to AUG too. Reference at `http://localhost:8012/docs`.

```bash
curl -s http://localhost:8012/chat/invoke \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"context_id": "laptop", "message": "Summarise https://news.ycombinator.com in three lines"}'
```

`/chat/stream` streams the turn as server-sent events. Add `"agent": "v15_codex"` to pick an agent.
