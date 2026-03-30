# 🦾 AUG 

<table>
  <tr>
    <td>

**Aug** is a self-hosted personal AI assistant that talks to you via Telegram or REST, ships with powerful tools, doesn't reveal your secrets to the LLM ([hushed](https://github.com/vadimtitov/hushed)), and has a persistent memory system that builds a real picture of you and itself across time. Works with any OpenAI-compatible API.

*Work in progress.*

  </td>
    <td><img src="https://github.com/user-attachments/assets/2cf5088d-433e-4ca5-b6e1-3f6fc1909b00" width="250"></td>
  </tr>
</table>


## Memory

The agent effectively writes its own system prompt. Five plain-text files on disk are injected into every conversation alongside hardcoded instructions — the agent reads them as context and writes back to them over time via background jobs and mid-conversation notes.

| File | What it is | Updated |
|------|------------|---------|
| `notes.md` | Raw notes captured mid-conversation, pending consolidation | During conversation |
| `user.md` | Who you are — biographical facts, traits, preferences, environment | Nightly |
| `context.md` | Current focus and recent activity. Volatile — stale entries trimmed freely | Nightly |
| `self.md` | The agent's identity: character, values, how it relates to you | Weekly |
| `memory.md` | Patterns and significant moments that have solidified across sessions | Weekly |

The `note` tool is the write mechanism during a conversation. A nightly job folds notes into `context.md` and `user.md`. A weekly deep consolidation promotes patterns into `memory.md` and — rarely — evolves `self.md`.



## Tools

| Tool | What it does |
|------|-------------|
| Web search | Search the web via [Brave Search API](https://brave.com/search/api/) |
| Browser | Control a real Chrome browser — log in, fill forms, navigate, screenshot, download files. Powered by [browser-use](https://github.com/browser-use/browser-use). See [setup guide](docs/browser-tool.md). |
| Shell | Run bash commands inside the container with secret injection and a command blocklist |
| Gmail | Search, read, send, and draft emails across multiple accounts. See [setup guide](docs/gmail-setup.md). |
| Image generation | Generate images from text (`IMAGE_GEN_MODEL`, default: gpt-image-1.5) |
| Image editing | Transform or edit an image you've sent |
| Files | Send files to the agent, it saves them to disk, processes via shell tools, and can send files back |
| Reminders | Set one-off or recurring reminders |
| Skills | User-defined persistent instructions following the [agentskills.io](https://agentskills.io/specification) spec. See [design doc](docs/skills-design.md). |
| Home Assistant | Control smart home devices via reflexes (runs in parallel with the agent) |
| Portainer | List, inspect, and manage Docker containers and stacks. See [setup guide](docs/portainer-setup.md). |


## Installation

**Portainer (intended for production)**

The primary deployment path. Add the repo in Portainer under *Stacks → Repository*, point it at `docker-compose.prod.yml`, and supply env vars via the Portainer UI. Portainer builds and runs the container directly from source — no registry needed. Assumes an external PostgreSQL database and a host directory at `/opt/aug/data` owned by UID 1000.

**Local / Docker**

Includes a Postgres container, good for development:

```bash
git clone https://github.com/vadimtitov/aug.git
cd aug
cp .env.example .env  # fill in API_KEY, LLM_*, DATABASE_URL
docker compose up --build
```

Interactive docs at `http://localhost:8000/docs`.



## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat/invoke` | Single-turn, full JSON response |
| `POST` | `/chat/stream` | SSE streaming response |
| `POST` | `/threads` | Create thread |
| `GET` | `/threads/{id}` | Thread metadata + history |
| `DELETE` | `/threads/{id}` | Delete thread |
| `POST` | `/files/upload` | Upload file |
| `GET` | `/files/{id}` | File metadata |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Yes | Shared secret for `X-API-Key` auth |
| `LLM_API_KEY` | Yes | API key for your LLM provider |
| `LLM_BASE_URL` | Yes | Any OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1` |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:password@host:5432/dbname` |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot; disabled if absent |
| `TELEGRAM_ALLOWED_CHAT_IDS` | No | Comma-separated allowed chat IDs. Unset = all chats allowed. Get yours from `@userinfobot`. |
| `BRAVE_API_KEY` | No | Enables web search |
| `IMAGE_GEN_MODEL` | No | Image generation model via LiteLLM (default: `gpt-image-1.5`) |
| `GMAIL_CLIENT_ID` | No | Gmail OAuth client ID (pair with `GMAIL_CLIENT_SECRET`) |
| `GMAIL_CLIENT_SECRET` | No | Gmail OAuth client secret |
| `BROWSER_CDP_URL` | No | Chrome DevTools Protocol URL for browser tool, e.g. `ws://chrome:9222` |
| `PORTAINER_URL` | No | Portainer instance URL |
| `PORTAINER_API_TOKEN` | No | Portainer API token |
| `HASS_URL` | No | Home Assistant URL |
| `HASS_TOKEN` | No | Home Assistant long-lived access token |
| `DEBUG` | No | `true` → human-readable logs; `false` (default) → JSON |


## Make targets

| Target | Description |
|--------|-------------|
| `make run` | Build and start, stream logs |
| `make rebuild` | Stop, rebuild, start |
| `make down` | Stop and remove containers |
| `make test` | Run pytest |
| `make lint` | ruff check |
| `make format` | ruff format |
| `make check` | Lint + format check + tests |
| `make logs` | Tail aug container logs |
| `make shell` | Shell into running aug container |
