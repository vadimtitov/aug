# Production Hardening Plan

## Tier 1 — High impact, low effort

### 1. Request / correlation ID
Middleware for REST API generates a UUID per HTTP request. For Telegram, a UUID is generated per incoming `Update`. Injected into every log line for the duration of that request or message — lets you grep all activity for a single conversation turn.

### 2. Asyncio timeout on LangGraph loops
`asyncio.timeout(300)` wrapping `astream_events` in the Telegram and REST interfaces. Prevents a runaway or stuck agent loop from hanging the event loop indefinitely.

### 3. Pydantic field constraints on API inputs
`max_length` and `pattern` validation on `message`, `thread_id`, and other free-text fields in request schemas. One-time change, permanently closes a class of oversized-payload and injection bugs.

### 4. Config cross-validation at startup
`model_validator` in `Settings` for paired fields — e.g. Gmail client ID without secret, Portainer URL without token. Raises at boot rather than silently failing at first tool call.

### 5. Non-root user in Docker
Run the container as a non-root user. Defense-in-depth: if `run_bash` is abused via prompt injection, the blast radius is limited. Costs one line in the Dockerfile.

### 6. Real DB probe in `/health`
Current health check returns `ok` unconditionally. Fix it to run `SELECT 1` against Postgres and return `degraded` if the DB is unreachable.

---

## Tier 2 — High impact, moderate effort

### 7. Ruff rule expansion
Add rule sets to `pyproject.toml`:
- `B` — flake8-bugbear (real bugs, not style)
- `ASYNC` — async anti-patterns (blocking calls in async context, missing awaits)
- `G` — no f-strings in log calls (defeats lazy formatting)
- `LOG` — logging best practices

### 8. Pre-commit hooks
`.pre-commit-config.yaml` with ruff check + ruff format. Enforces quality automatically on every commit, no manual `make check` needed.

### 9. Graceful shutdown
- `stop_grace_period: 30s` in `docker-compose.prod.yml`
- `--timeout-graceful-shutdown 25` added to uvicorn command
- Clean `CancelledError` handling in the reminder loop (log shutdown, don't swallow)
- Consolidation scheduler task cancelled in lifespan teardown (currently may leak)

### 10. Dead letter for reminders
Add `max_retries` (e.g. 50) and `dead_lettered_at` columns to the reminders table. Reminders that exhaust retries are marked dead and logged as errors rather than retrying forever at 60-minute intervals.

### 11. asyncpg pool tuning
Explicit `min_size=2`, `max_size=15`, `command_timeout=30` on pool creation. Fail-fast DB connectivity check at startup — raise immediately if Postgres is unreachable rather than failing on first request.

---

## Tier 3 — Medium impact

### 12. Background task watchdog
Every time the reminder loop completes a cycle, write `app.state.last_reminder_check = now`. The `/health` endpoint checks if the timestamp is stale (> 2 min) and returns `degraded`. Surfaces silently stuck background loops before users notice missed reminders.

### 13. Token usage logging
Log `total_tokens` from every LLM response (`AIMessage.response_metadata`). Zero overhead, immediate cost visibility in logs.

### 14. Startup config summary log
Log effective configuration at boot with secrets redacted (e.g. `telegram_enabled: true`, `brave_search_enabled: false`). Invaluable when debugging a prod misconfiguration.

---

## Deferred

- **Prometheus / Grafana** — worth adding when there's real traffic to observe
- **Langfuse / LiteLLM cost tracking** — revisit when per-conversation cost breakdown becomes useful
- **OpenTelemetry tracing** — Langfuse covers LLM-specific tracing when needed
- **mypy / pyright** — LangGraph typing is rough; not worth the noise yet
- **Zero-downtime deployment** — 5-second gap on a personal assistant is acceptable
