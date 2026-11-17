# Configuration

AUG reads environment variables at startup: from `.env` with the installer or Compose, or from Portainer's environment settings. Behaviour you change while it runs (agent per chat, approval rules, the bash blocklist) lives in `data/settings.json`. You can edit that file through the mini app or by hand.

## Required

| Variable | |
|---|---|
| `API_KEY` | The REST API key (`X-API-Key`). Any long random string |
| `DATABASE_URL` | `postgresql+asyncpg://user:password@host:5432/db`. Set for you by `docker-compose.standalone.yml` |
| `LLM_BASE_URL` | An OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1` or a LiteLLM proxy. See [what uses it](#the-openai-compatible-endpoint) |
| `LLM_API_KEY` | Its key. May be empty if you use only Claude Code and Codex |
| `BASE_URL` | Where links AUG sends you point (account sign-ins, the mini app). Must be `https://`, or `http://localhost` / `http://127.0.0.1` |

## Telegram

| Variable | |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather. Without it, the bot is off and the REST API still works |
| `TELEGRAM_ALLOWED_CHAT_IDS` | Comma-separated Telegram user ids allowed to talk to the bot. **Empty means everyone.** Your id: @userinfobot |
| `STARTUP_ANNOUNCEMENT` | `true` (default): send "AUG is up" to your chats at each start |

## Optional integrations

| Variable | |
|---|---|
| `OAUTH_ENCRYPTION_KEY` | 32 random bytes, base64 (`openssl rand -base64 32`). Encrypts stored OAuth tokens. Lose it and every connection must be made again |
| `BRAVE_API_KEY` | Web search for the Kimi and GLM agents. Claude Code and Codex search on their own |
| `STEALTH_BROWSER_URL` | The stealth browser sidecar, e.g. `http://stealth-browser:8100`. Without it, pages blocked by bot protection stay blocked |
| `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET` | Gmail ([setup](setup-guides/gmail-setup.md)). Set both or neither |
| `HASS_URL`, `HASS_TOKEN` | Home Assistant, for the `v15_*` reflex |
| `PORTAINER_URL`, `PORTAINER_API_TOKEN` | Portainer tools ([setup](setup-guides/portainer-setup.md)). Set both or neither |
| `AUG_CONTAINER`, `AUG_ENVIRONMENT` | AUG's own container and Portainer environment, so installing an MCP server can restart AUG for you |
| `BROWSER_CDP_URL` | A Chromium DevTools endpoint: the mini app's live browser view, and the Playwright MCP server |
| `IMAGE_GEN_MODEL` | Image model on the endpoint (default `gpt-image-1.5`) |
| `MINIAPP_API_URL` | The API's public URL, baked into the mini app at build time |
| `CHROME_USER`, `CHROME_PASSWORD` | Basic auth for the Chromium web UI in the production stack |
| `DEBUG` | `true`: plain-text, verbose logs, and any `BASE_URL` is accepted |

The installer's commands read two more: `AUG_PORT` (the local API port, default 8012) and `AUG_DIR` (where `install.sh` clones, default `~/aug`).

Never set `DEV_AUTH_BYPASS` outside development. It turns off the mini app's Telegram sign-in check.

## The OpenAI-compatible endpoint

Claude Code and Codex never use `LLM_BASE_URL`. These do:

| Feature | Model it asks for |
|---|---|
| Voice note transcription | `whisper-1` |
| Image generation and editing | `IMAGE_GEN_MODEL` |
| The Kimi and GLM agents | `accounts/fireworks/models/kimi-k3`, `glm-5p3`, `glm-5p3-flash` |
| Memory consolidation | runs on `v13_kimi` |
| Home Assistant reflex | `gpt-6-luna` (OpenAI's Decisions API) |

An OpenAI key covers the first two. For all of them, run [LiteLLM](https://github.com/BerriAI/litellm) with OpenAI and Fireworks behind it, and point `LLM_BASE_URL` at it.

## Secrets

The compose files load the API, database, Telegram and other keys into [hushed](https://github.com/vadimtitov/hushed) at each start, and run AUG under `hushed run`. Add your own with `/secret` in Telegram or `./augctl secret NAME`. The agent sees their names and can use them as `$NAME` in commands; their values are redacted from all output.

## settings.json

| Key | |
|---|---|
| `conversations.<id>.agent` | The agent per chat (what `/version` writes) |
| `tools.approvals` | Saved approval rules (what `/approvals` lists) |
| `tools.bash.blacklist` | Extra regex patterns the agent's shell commands may not match |
| `reflexes.homeassistant.entity_label` | The Home Assistant label for devices the reflex may switch (default `aug`) |
