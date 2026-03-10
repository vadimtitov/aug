# AUG — Agent Using Graph

Personal AI assistant backend powered by **FastAPI** and **LangGraph 1.0**.

AUG is designed for personal use with multiple frontends: a web UI, iOS app, and Telegram bot. All agent logic runs as a LangGraph state-machine; all LLM calls are routed through a LiteLLM proxy.

---

## Quick start

```bash
cp .env.example .env   # fill in API_KEY, LLM_*, DATABASE_URL
make dev               # local uvicorn with hot-reload
```

Or with Docker (includes Postgres):

```bash
make run
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/chat/invoke` | Full JSON response |
| POST | `/chat/stream` | SSE streaming response |
| POST | `/threads` | Create thread |
| GET | `/threads/{id}` | Thread metadata + history |
| DELETE | `/threads/{id}` | Delete thread |
| POST | `/files/upload` | Upload file |
| GET | `/files/{id}` | File metadata |
| POST | `/telegram/webhook` | Telegram webhook (optional) |

Interactive docs: `http://localhost:8000/docs`

## Project layout

```
aug/                  ← Python package
  app.py              ← FastAPI factory + lifespan
  config.py           ← pydantic-settings
  core/               ← LangGraph graphs, agents, state
  api/                ← Routers, schemas, security, Telegram
  utils/              ← DB pool, file storage, logging
tests/
Makefile
pyproject.toml
Dockerfile
docker-compose.yml
docker-compose.prod.yml
```

## Tech stack

- **FastAPI** + **uvicorn**
- **LangGraph 1.0** (StateGraph, no langgraph-cli dependency)
- **LangChain / ChatOpenAI** → LiteLLM proxy
- **asyncpg** + **langgraph-checkpoint-postgres**
- **pydantic-settings**, **ruff**, **uv**, **pytest**
- **python-telegram-bot** (optional, webhook mode)

## Make targets

```
make dev      # local hot-reload
make run      # docker compose up --build
make down     # docker compose down
make test     # pytest
make lint     # ruff check
make format   # ruff format
make logs     # docker compose logs -f aug
make shell    # exec into aug container
```
