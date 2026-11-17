# Using AUG

- [Telegram](#telegram)
- [Memory and persona](#memory-and-persona)
- [Skills](#skills)
- [Tasks and reminders](#tasks-and-reminders)
- [Tools](#tools)
- [REST API](#rest-api)

## Telegram

Talk to the bot as you would to a person. You can send:

- **Text.** A message sent while the agent is working joins the turn in progress.
- **Voice notes.** They're transcribed, so this needs `LLM_API_KEY` with an OpenAI key.
- **Photos, documents, video.** They're saved to the workspace, and the agent opens them with its tools.
- **Locations.** One-off and live. Live updates are throttled (`/throttle`).

The agent replies with text and files, and can message you first: when a scheduled task runs, or when background work finishes after its turn ended.

### Commands

| Command | |
|---|---|
| `/clear` | Start a new conversation in this chat |
| `/stop` | Stop the current turn |
| `/status` | Model, context usage, plan limits, background work |
| `/compact` | Summarise the conversation to free context |
| `/version` | Switch this chat's agent |
| `/claude_login`, `/codex_login` | Sign the Claude or Codex agents in (private chat) |
| `/cancel` | End a sign-in in progress |
| `/secret` | Store a secret in hushed. The message holding it is deleted |
| `/skills` | Browse the skills and their files |
| `/prompt` | The current session's system prompt, as a file |
| `/consolidate`, `/consolidate_deep` | Run the nightly or weekly memory pass now |
| `/ssh` | Add or remove SSH targets |
| `/approvals` | Review and revoke saved approvals |
| `/context_id` | This chat's session id (for the REST API) |
| `/throttle` | How often live location updates reach the agent |

### Forum topics

For several parallel conversations, make a private group with topics turned on and add the bot. Each topic is its own chat with its own agent; a topic with none picked inherits the group's. [Setup](setup-guides/telegram-forum-setup.md).

## Memory and persona

Everything the assistant is and knows lives in `data/workspace/`. These are plain files, and you can edit them.

| | |
|---|---|
| `AGENTS.md` | Persona and approach. Yours to rewrite; the agent changes it only when you ask |
| `memory/MEMORY.md` | The index, one line per memory file. It's in every session's prompt |
| `memory/notes.md` | Notes the agent takes as you talk (`note` tool), waiting to be consolidated |
| `memory/user.md` | Who you are: facts, preferences, rules |
| `memory/context.md` | What's going on now: plans, events, recent activity. Pruned hard |
| `memory/self.md` | The assistant's sense of itself |

Every night a consolidation pass (the "dream") folds the notes into the other files. Once a week a deeper pass looks across all of them. Both run on the default agent, through your Claude plan.

A session's prompt is built once, when it starts, and stays fixed for the whole conversation, so the prompt cache holds. Edits show up from the next `/clear`. [Memory design](design/memory-design.md).

## Skills

Skills follow the [agentskills.io](https://agentskills.io/specification) format: a folder with a `SKILL.md` and optional scripts, references and assets, under `data/workspace/skills/`. The agent writes them when you ask ("make a skill for filing my receipts"), and loads one when a task calls for it. On Claude Code and Codex they load as the CLI's own skills.

A Python script in a skill can declare its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)). AUG pre-fetches them at startup. [Skills design](design/skills-design.md).

## Tasks and reminders

"Remind me at 9 tomorrow to call the bank." "Every Monday at 8, summarise my unread email." The agent schedules these with `create_task`: one-off or recurring (cron, interval). When a task fires, it either sends you the text as written or runs the agent with it, in the chat it was made in. `list_tasks`, `update_task` and `delete_task` manage them.

External systems can push in too: `POST /hooks/push` relays a message to a chat, or runs the agent with it ([webhook design](design/webhook-design.md)).

## Tools

All agents have these, on top of what Claude Code or Codex bring themselves:

| | |
|---|---|
| **Web** | `web_fetch` reads any page and answers a question about it. When a site blocks plain requests, a stealth browser (Camoufox) takes over. [Evaluation](design/web-fetching-evaluation.md) |
| **Files** | Files you send land in the workspace; `respond_with_file` sends files back |
| **Memory** | `note` |
| **Tasks** | `create_task`, `list_tasks`, `update_task`, `delete_task` |
| **Skills** | `save_skill`, `write_skill_file`, `delete_skill` |
| **SSH** | Commands and file transfer on hosts you add with `/ssh`. AUG makes the key, installs it and pins the host fingerprint. Every command needs your approval, unless a saved rule covers it |
| **Gmail** | Search, read, draft, send, across several accounts. [Setup](setup-guides/gmail-setup.md) |
| **Images** | Generate and edit images (needs an OpenAI key) |
| **Portainer** | Containers and stacks: list, logs, start/stop, deploy. [Setup](setup-guides/portainer-setup.md) |
| **MCP** | The agent searches the MCP registry, installs servers and removes them. Their tools join every agent after a restart |

The container also has Python with pandas and matplotlib, Node, ffmpeg, pandoc, ImageMagick, Tesseract, poppler, exiftool, jq and the GitHub CLI.

**Secrets.** Store an API key with `/secret` or `./augctl secret NAME`. The agent can use it in a shell command (`$NAME`), but the value is redacted from everything it reads back. [Security](security.md).

## REST API

Everything Telegram does is also an API, authenticated with `X-API-Key: <API_KEY>` (it's in `.env`). The interactive reference is at `http://localhost:8012/docs`, or on whatever `AUG_PORT` you chose.

```bash
curl -s http://localhost:8012/chat/invoke \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"context_id": "laptop", "message": "Summarise https://news.ycombinator.com in three lines"}'
```

| | |
|---|---|
| `POST /chat/invoke` | One turn, a JSON reply |
| `POST /chat/stream` | One turn as server-sent events: text, tool calls, files, approvals |
| `POST /chat/{context_id}/run/cancel` | Stop the turn |
| `GET /chat/{context_id}/status` | Model, context, limits |
| `GET /chat/{context_id}/approvals`, `POST …/approve` | Answer approval requests |
| `POST /files/upload` | Upload a file to attach to a message |
| `POST /threads`, `GET`/`DELETE /threads/{context_id}` | Create, read and delete conversations |
| `POST /hooks/push` | Push a message into a chat |
| `GET /health` | Liveness and database check |

Pass `"agent": "v15_codex"` to pick an agent per request. Without it, the chat's own agent is used.
