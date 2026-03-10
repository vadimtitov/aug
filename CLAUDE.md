# CLAUDE.md — AUG Development Guide

## What is AUG?

**AUG** stands for **Agent Using Graph**. It is a personal AI assistant backend:
- FastAPI application, designed for personal use
- Agent logic implemented as LangGraph 1.0 state-machines
- Frontends: web UI, iOS app, Telegram bot
- All LLM calls routed through a LiteLLM proxy

---

## Project structure

```
aug/                          ← repo root
├── aug/                      ← Python package (the application)
│   ├── app.py                ← FastAPI factory + lifespan (DB pool, checkpointer, Telegram)
│   ├── config.py             ← All settings via pydantic-settings; reads .env
│   ├── core/
│   │   ├── llm.py            ← LLM factory — always ChatOpenAI → LiteLLM proxy
│   │   ├── state.py          ← AgentState TypedDict shared by all graphs
│   │   ├── prompts.py        ← System prompts per agent; edit prompts here
│   │   ├── graph.py          ← Agent registry + compiled-graph cache
│   │   └── agents/
│   │       └── default.py    ← Default agent (POC: hardcoded response)
│   ├── api/
│   │   ├── security.py       ← require_api_key FastAPI dependency
│   │   ├── schemas/          ← Pydantic request/response models
│   │   ├── routers/          ← FastAPI routers (chat, threads, files)
│   │   └── telegram.py       ← Telegram webhook; optional at startup
│   └── utils/
│       ├── db.py             ← asyncpg pool + schema bootstrap
│       ├── storage.py        ← File storage abstraction (LocalFileStorage default)
│       └── logging.py        ← JSON (prod) or human-readable (dev) structured logging
├── tests/
│   └── test_health.py        ← Smoke test (mocks DB + checkpointer)
├── Makefile                  ← dev / run / test / lint / format targets
├── pyproject.toml            ← uv dependencies + ruff + pytest config
├── Dockerfile                ← python:3.12-slim, uv, non-root user, /health check
├── docker-compose.yml        ← local dev: aug + postgres:16
└── docker-compose.prod.yml   ← production: aug only, external postgres network
```

---

## Running locally

```bash
cp .env.example .env   # fill in the required values
make dev               # uvicorn with --reload on port 8000
```

Postgres must be running and `DATABASE_URL` must be set. For a full local stack:

```bash
make run   # docker compose up --build (starts Postgres too)
```

---

## Adding a new agent

1. Create `aug/core/agents/<name>.py`:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from aug.core.state import AgentState

AGENT_CONFIG = {
    "model": "gpt-4o",   # any string LiteLLM understands
    "temperature": 0.5,
}

def build(checkpointer: BaseCheckpointSaver) -> CompiledGraph:
    graph = StateGraph(AgentState)
    # ... add nodes and edges ...
    return graph.compile(checkpointer=checkpointer)
```

2. Register it in `aug/core/graph.py`:

```python
_REGISTRY: dict[str, str] = {
    "default": "aug.core.agents.default",
    "researcher": "aug.core.agents.researcher",  # add this line
}
```

3. Redeploy. No env var changes needed.

---

## Adding a new tool

1. Create `aug/core/tools/<name>.py` exporting a LangChain `@tool`-decorated function.
2. Import it in the relevant agent file and pass it to the LLM or graph node.

Example:

```python
# aug/core/tools/web_search.py
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web and return a summary."""
    ...
```

---

## Key architectural rules

### No LangSmith dependency
Do not use `langgraph up`, `langgraph-cli`, or any LangSmith tooling. The graph
is a plain Python object instantiated inside FastAPI. No `LANGSMITH_API_KEY` required.

### Thread–agent locking
A thread is permanently bound to the agent version used to create it.
- `POST /threads` stores `agent_version` in Postgres.
- `POST /chat/invoke` and `POST /chat/stream` must use the same agent version.
- Mismatch → HTTP 400.

### LLM access — always via `core/llm.py`
All LLM calls must go through `aug.core.llm.build_llm(config)`.
- Always returns a `ChatOpenAI` instance pointed at `LLM_BASE_URL` (LiteLLM proxy).
- Never instantiate `ChatOpenAI` (or any other provider class) directly in agent files.

### Model selection is per-agent
Each agent file contains its own `AGENT_CONFIG = {"model": "...", ...}`.
To change the model for an agent, edit that file. Do not add model env vars.

### API authentication
All routes except `GET /health` and `POST /telegram/webhook` require:
```
X-API-Key: <API_KEY>
```
Applied at the router level in `api/security.py`.

### Async everywhere
Use `async def` throughout — FastAPI handlers, asyncpg queries, LangGraph invocation.

### ruff only
Do not add black, flake8, isort, or any other linter/formatter. ruff covers all of them.

### uv only
All dependency management is via `pyproject.toml` + `uv`. Do not generate `requirements.txt`.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | Shared secret for `X-API-Key` auth |
| `LLM_API_KEY` | Yes | Passed to the OpenAI client (can be a dummy string if LiteLLM handles auth) |
| `LLM_BASE_URL` | Yes | LiteLLM proxy URL, e.g. `http://litellm:4000` |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:password@host:5432/dbname` |
| `TELEGRAM_BOT_TOKEN` | No | Bot disabled if absent |
| `TELEGRAM_WEBHOOK_SECRET` | No | Validates `X-Telegram-Bot-Api-Secret-Token` header |
| `TELEGRAM_WEBHOOK_URL` | No | Public URL used for webhook registration |
| `DEBUG` | No | `true` → human-readable logs; `false` (default) → JSON logs |
