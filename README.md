# 🦾 AUG 

<table>
  <tr>
    <td>

**Aug** is a self-hosted personal AI assistant that talks to you via Telegram or REST, ships with powerful tools, doesn't reveal your secrets to the LLM ([hushed](https://github.com/vadimtitov/hushed)), and has a persistent memory system that builds a real picture of you and self across time. Works with any OpenAI-compatible API.

*Work in progress.*

  </td>
    <td><img src="https://github.com/user-attachments/assets/2cf5088d-433e-4ca5-b6e1-3f6fc1909b00" width="250"></td>
  </tr>
</table>


## Memory

Three plain-text files on disk, injected into the system prompt at runtime, updated by background jobs. No embeddings, no retrieval — everything fits in context.

| File | What it is |
|------|------------|
| `self.md` | The agent's identity: character, values, how it relates to you. Changes rarely — only through weekly reflection, never mid-conversation. |
| `user.md` | Who you are. Biographical facts, core traits. Slow-moving by design. |
| `memory.md` | Everything else: *Present*, *Recent*, *Patterns*, *Significant moments*, *Reflections*, *Longer arc*. |

Mid-conversation, the agent uses the `note` tool to capture things worth remembering. A nightly job folds notes into `memory.md`. A weekly deep consolidation compresses *Recent* into *Patterns*, updates the longer arc, and — rarely — evolves `self.md`.

See [docs/memory-design.md](docs/memory-design.md) for the full design rationale.


## Installation

**Portainer (intended for production)**

The primary deployment path. Add the repo in Portainer under *Stacks → Repository*, point it at `docker-compose.prod.yml`, and supply env vars via the Portainer UI. Portainer builds and runs the container directly from source — no registry needed. Assumes an external PostgreSQL database and a host directory at `/opt/aug/data` owned by UID 1000.

**Local / Docker**

Includes a Postgres container, good for development:

```bash
git clone https://github.com/your-username/aug.git
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
| `DEBUG` | No | `true` → human-readable logs; `false` (default) → JSON |


## Make targets

| Target | Description |
|--------|-------------|
| `make run` | `docker compose up --build` |
| `make down` | Stop and remove containers |
| `make test` | Run pytest |
| `make lint` | ruff check |
| `make format` | ruff format |
| `make check` | Lint + format check + tests |
| `make logs` | Tail aug container logs |
| `make shell` | Shell into running aug container |
