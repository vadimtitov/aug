# CLAUDE.md — AUG Development Guide

## What is AUG?

**AUG** — Agent Using Graph. Personal AI assistant backend:
- FastAPI + LangGraph 1.0 state machines
- All LLM calls routed through a LiteLLM proxy
- Frontends: Telegram bot, REST API (web/iOS)

---

## Project structure

```
aug/
├── aug/
│   ├── app.py                  ← FastAPI factory + lifespan (DB pool, checkpointer, Telegram)
│   ├── config.py               ← Settings via pydantic-settings; get_settings() is lazy + cached
│   ├── core/
│   │   ├── llm.py              ← build_chat_model() factory — always ChatOpenAI → LiteLLM proxy
│   │   ├── state.py            ← AgentState + AgentStateUpdate (Pydantic models)
│   │   ├── prompts.py          ← get_user_context() — assembles runtime system prompt context
│   │   ├── registry.py         ← Agent registry + compiled-graph cache
│   │   └── agents/
│   │       ├── base_agent.py   ← BaseAgent ABC with agentic loop in build()
│   │       ├── chat_agent.py   ← ChatAgent (configurable) + TimeAwareChatAgent
│   │       └── fake_agent.py   ← Hardcoded response, no LLM (for testing)
│   ├── api/
│   │   ├── security.py         ← require_api_key FastAPI dependency
│   │   ├── schemas/            ← Pydantic request/response models
│   │   ├── routers/            ← FastAPI routers (chat, threads, files)
│   │   └── telegram.py         ← Telegram polling bot
│   └── utils/
│       ├── db.py               ← asyncpg pool + schema bootstrap
│       ├── storage.py          ← File storage abstraction
│       ├── logging.py          ← JSON (prod) / human-readable (dev) structured logging
│       ├── data.py             ← read_data_file() / write_data_file() — /app/data volume access
│       └── user_settings.py    ← per-user settings; get_setting() / set_setting() with nested path
├── tests/
├── Makefile
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml          ← local dev: aug + postgres
└── docker-compose.prod.yml     ← production: aug only
```

---

## Running locally

```bash
cp .env.example .env   # fill in required values
make run               # docker compose up --build — logs stream to terminal, Ctrl+C stops
```

---

## Testing the agent via API

Local API key is `123` (set in `.env`).

**Single invoke:**
```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 123" \
  -d '{"message": "what time is it?", "thread_id": "test-1", "agent": "default"}' \
  | python3 -m json.tool
```

**Streaming (SSE):**
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 123" \
  -d '{"message": "search the web for latest AI news", "thread_id": "test-2", "agent": "default"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## Adding a new agent

1. Create `aug/core/agents/<name>_agent.py` subclassing `BaseAgent` or `ChatAgent`.
2. Register an instance in `aug/core/registry.py`.

```python
# aug/core/registry.py
_REGISTRY: dict[str, BaseAgent] = {
    "default": TimeAwareChatAgent(
        model="gpt-4o",
        system_prompt="You are ...",
        tools=[brave_search],
        temperature=0.8,
    ),
    "my_agent": ChatAgent(model="gpt-4o-mini", system_prompt="..."),
}
```

`ChatAgent` accepts: `model`, `system_prompt`, `tools`, `temperature`, `max_tokens`,
`max_retries`, `timeout`, `seed`.

### Hooks (override in subclass)
- `preprocess(state)` — runs before LLM; inject system prompt, annotate messages, etc.
- `respond(state)` — **required**; calls the LLM.
- `postprocess(state)` — runs after LLM; logging, filtering, etc.

All return `AgentStateUpdate(messages=[...])` or `AgentStateUpdate(system_prompt="...")`.

---

## Adding a new tool

1. Create `aug/core/tools/<name>.py` with a `@tool`-decorated function.
2. Add it to the relevant agent in `registry.py`.

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Description the LLM uses to decide when to call this tool."""
    ...
```

---

## User settings

Per-user settings are stored in `/app/data/settings.json` via `aug/utils/user_settings.py`.

Schema: top-level key is the **interface namespace**, then sub-keys, then entity ID, then the setting:

```json
{
  "telegram": {
    "chats": {
      "<chat_id>": { "agent": "default" }
    }
  }
}
```

API:
```python
get_setting("telegram", "chats", str(chat_id), "agent", default="default")
set_setting("telegram", "chats", str(chat_id), "agent", value="v1_claude")
```

Rules:
- Always namespace by interface (`"telegram"`, `"api"`, etc.) — entity IDs differ per interface and must not be mixed.
- Sub-keys within an interface group related features (e.g. `"chats"` for per-chat config).
- Do not add a flat `"agent"` directly under `"telegram"` — that level is reserved for interface-wide settings.

---

## Key rules

- **LLM access** — always via `build_chat_model()` in `aug/core/llm.py`. Never instantiate `ChatOpenAI` directly.
- **Settings** — always via `get_settings()` from `aug/config.py`. Never import `settings` at module level.
- **Async everywhere** — `async def` throughout.
- **ruff only** — no black, flake8, isort.
- **uv only** — no `requirements.txt`.

## Code style

- **All imports at the top of the file** — no inline imports inside functions. Solve circular imports by restructuring modules, not by deferring imports.
- **Private functions at the bottom** — public API first, helpers (`_prefixed`) last.
- **No unnecessary abstractions** — three similar lines beat a premature helper.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | Shared secret for `X-API-Key` auth |
| `LLM_API_KEY` | Yes | Passed to OpenAI client (dummy string if LiteLLM handles auth) |
| `LLM_BASE_URL` | Yes | LiteLLM proxy URL, e.g. `http://litellm:4000` |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:password@host:5432/dbname` |
| `TELEGRAM_BOT_TOKEN` | No | Bot disabled if absent |
| `TELEGRAM_ALLOWED_CHAT_IDS` | No | Comma-separated chat IDs allowed to use the bot. If unset, all chats are allowed. Get your ID from `@userinfobot` |
| `BRAVE_API_KEY` | No | Web search tool disabled if absent |
| `DEBUG` | No | `true` → human-readable logs; `false` (default) → JSON |
