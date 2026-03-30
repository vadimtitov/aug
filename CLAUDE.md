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
│   │   ├── prompts.py          ← build_system_prompt() — assembles full system prompt from memory files + state
│   │   ├── memory.py           ← Memory file init, append_note(), light + deep consolidation, scheduler
│   │   ├── registry.py         ← Agent registry + compiled-graph cache
│   │   ├── agents/
│   │   │   ├── base_agent.py   ← BaseAgent ABC with agentic loop in build()
│   │   │   ├── chat_agent.py   ← ChatAgent, TimeAwareChatAgent, AugAgent
│   │   │   └── fake_agent.py   ← Hardcoded response, no LLM (for testing)
│   │   └── tools/
│   │       ├── brave_search.py ← Web search via Brave API
│   │       ├── fetch_page.py   ← URL content extraction
│   │       ├── run_bash.py     ← Shell access with secret injection + blocklist
│   │       ├── note.py         ← Mid-conversation memory capture
│   │       └── memory.py       ← Legacy key-value memory tools (remember/recall/forget)
│   ├── api/
│   │   ├── security.py         ← require_api_key FastAPI dependency
│   │   ├── schemas/            ← Pydantic request/response models
│   │   ├── routers/            ← FastAPI routers (chat, threads, files)
│   │   └── telegram.py         ← Telegram polling bot
│   └── utils/
│       ├── db.py               ← asyncpg pool + schema bootstrap
│       ├── storage.py          ← File storage abstraction
│       ├── logging.py          ← JSON (prod) / human-readable (dev) structured logging
│       ├── data.py             ← DATA_DIR, MEMORY_DIR, read_data_file() / write_data_file()
│       ├── homeassistant.py    ← HomeAssistantClient: typed HA REST+WebSocket client with entity cache
│       └── user_settings.py    ← per-user settings; get_setting() / set_setting() with nested path
├── tests/
├── docs/
│   ├── memory-design.md        ← Memory system design rationale
│   └── TODO.md                 ← Open tasks
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

Local API key is set in `.env` as `API_KEY`.

**Single invoke:**
```bash
curl -s -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "what time is it?", "thread_id": "test-1", "agent": "default"}' \
  | python3 -m json.tool
```

**Streaming (SSE):**
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
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

### Tool implementation standard

The user does not have time to manually test every tool. **You are expected to test it yourself** before considering a tool done. The service is running locally, the API key is in `API_KEY` env var, and you know how to call the endpoints. Use them.

**What "done" means for a tool:**

1. **It actually works end-to-end.** Call the streaming endpoint with a real prompt that exercises the tool. Watch the logs. Confirm the tool ran and returned a sensible result — not just that the code looks right.

2. **Failure modes are handled honestly.** When a tool fails, it must return an unambiguous error string. Never return a neutral or success-sounding message when the task didn't complete — the main agent will take it at face value and lie to the user. If `final_result` is None, if a subprocess failed, if an API returned an error — say so explicitly. A good failure string directly tells the agent what to report: `"Task did NOT complete. Errors: ..."`.

3. **The return value is informative.** The tool's return value is what the main LLM reads. Vague strings like `"Task completed."` or `"Done."` cause hallucination. Be specific about what was done, what was found, or what failed.

4. **Third-party library APIs are verified.** Don't assume the library's API matches its documentation or your prior knowledge. Read the installed source (`site-packages`) to confirm method signatures, parameter names, and return types before using them. Libraries change.

5. **Unit tests cover the real behaviour.** Tests must mock at the right boundary. If a function does DNS resolution, mock the DNS call. If a return value changed shape (e.g. resolved hostname in URL), update the assertion. Tests that pass by accident are worse than no tests.

6. **Corner cases are tested:**
   - Tool not configured (missing env var, missing API key) → graceful message
   - External service fails or times out → clean error, no crash
   - Empty / None return from underlying library → handled explicitly
   - Ambiguous success (task ran but produced no result) → treated as failure, not success

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

## Configuration philosophy

Making behaviour configurable via `settings.json` is good — but only when we are
confident a setting will be used and won't need to change shape later. Once something
is in `settings.json`, it is part of the interface: users may set it, scripts may
depend on it, and migrating or removing it has a cost.

Before adding a new setting, ask: is this actually going to be changed, or is it
just a hardcoded value with extra steps? Default to hardcoding. Promote to config
only when there is a clear, immediate reason.

## Key rules

- **LLM access** — always via `build_chat_model()` in `aug/core/llm.py`. Never instantiate `ChatOpenAI` directly.
- **Settings** — always via `get_settings()` from `aug/config.py`. Never import `settings` at module level.
- **Prompts** — all hardcoded strings passed to an LLM (system prompts, tool instructions, consolidation prompts, format instructions) must be defined as named constants in `aug/core/prompts.py`. Never define prompt strings inline in other modules.
- **Async everywhere** — `async def` throughout.
- **ruff only** — no black, flake8, isort.
- **uv only** — no `requirements.txt`.
- **Agent versions are immutable** — never modify the tools or behaviour of an existing agent version. Add a new version instead (`v3_claude`, `v4_claude`, etc.).

## Code style

- **All imports at the top of the file** — no inline imports inside functions. Solve circular imports by restructuring modules, not by deferring imports.
- **Private functions at the bottom** — public API first, helpers (`_prefixed`) last.
- **No unnecessary abstractions** — three similar lines beat a premature helper.

## Module and class design

- **Separate concerns by layer.** Client code (how to talk to an API) belongs in `utils/`. Orchestration code (what to do with the data) belongs in `core/`. When both live in the same file, neither is clean. See `aug/utils/homeassistant.py` + `aug/core/reflexes/homeassistant.py` as the canonical example.

- **State belongs to objects, not modules.** If you find yourself writing `_thing: SomeType`, `_thing_at: float`, and `_thing_lock: asyncio.Lock | None` at module level — stop. Those are instance variables. Create a class.

- **Use types everywhere.** Named dataclasses (`@dataclass(frozen=True)`) instead of raw `dict`. Typed method signatures. Makes code self-documenting and lets the type checker catch mistakes.

- **The public interface should read like English.** `client.get_entities(label="aug")` is obvious. Ten private helpers with interleaved global state are not. If you have to explain what a function does, the abstraction is wrong.

- **Ask "what layer does this belong to?" before writing.** Every function should have an obvious home. If you're hesitating, the module boundary is probably wrong.

- **The smell test:** if you see scattered `_foo`, `_foo_at`, `_foo_lock` globals — that's a missing class. If you see client logic mixed with orchestration logic in one file — that's a missing module split.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | Shared secret for `X-API-Key` auth |
| `LLM_API_KEY` | Yes | Passed to OpenAI client (dummy string if LiteLLM handles auth) |
| `LLM_BASE_URL` | Yes | LiteLLM proxy URL, e.g. `http://litellm:4000` |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host:5432/dbname` |
| `TELEGRAM_BOT_TOKEN` | No | Bot disabled if absent |
| `TELEGRAM_ALLOWED_CHAT_IDS` | No | Comma-separated chat IDs allowed to use the bot. If unset, all chats are allowed. Get your ID from `@userinfobot` |
| `BRAVE_API_KEY` | No | Web search tool disabled if absent |
| `IMAGE_GEN_MODEL` | No | Image generation model via LiteLLM (default: `gpt-image-1.5`) |
| `PORTAINER_URL` | No | Portainer instance URL, e.g. `http://portainer:9000` |
| `PORTAINER_API_TOKEN` | No | Portainer API token (generate in Portainer → Account Settings) |
| `PORTAINER_ENDPOINT_ID` | No | Portainer environment ID (default: `1`) |
| `DEBUG` | No | `true` → human-readable logs; `false` (default) → JSON |
