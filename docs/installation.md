# Installation

AUG runs in Docker. The container is the agent's sandbox: Claude Code and Codex get a full shell inside it, with no approval prompts. That's why there is no native install. On your own machine, the agent would have your whole home directory.

You need:

- **Docker** with Compose v2. On a Mac that means Docker Desktop, OrbStack or Colima. The first build takes 5–15 minutes.
- **A Claude or ChatGPT plan** for the Claude Code or Codex agents (Pro, Max, Plus, Team…).
- **A Telegram bot** (optional but recommended): message [@BotFather](https://t.me/BotFather), send `/newbot`, copy the token.

| You want | Use |
|---|---|
| To try it on your laptop, or run it on a home server | [The installer](#the-installer) |
| To pick every setting yourself | [Manual setup](#manual-setup) |
| Portainer, an external Postgres, the Telegram mini app | [Portainer and production](#portainer-and-production) |

## The installer

```bash
curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
```

It clones AUG into `~/aug` (set `AUG_DIR` to change that) and runs `./augctl setup`, which:

1. **Asks for your bot token**, then asks you to message the bot once. That's how it learns your Telegram user id. Only that account can use the bot.
2. **Asks for an OpenAI API key** (optional). It's used for voice notes and image generation; leave it empty to skip both.
3. **Writes `.env`** with random keys for the API and token encryption. The file is readable by you only.
4. **Builds and starts** three containers: the app, Postgres and the stealth browser.
5. **Signs you in** to Claude, Codex or both ([below](#signing-in)).

Then open your bot in Telegram and say hello.

Prefer to read scripts before you run them? Clone the repo and run `./augctl setup` from it. It does the same thing.

### Signing in

```bash
./augctl login claude    # Claude Code agents
./augctl login codex     # Codex agents
```

**Claude.** If Claude Code is installed on your machine, run `claude setup-token` there and paste the token it prints. Otherwise press Enter: AUG shows a sign-in link, you paste the code back, and Claude prints the token for you to paste once more. Either way it's valid for a year and goes straight into AUG's secret store.

**Codex.** Open the link, sign in to ChatGPT and enter the short code shown. Codex refreshes this login on its own.

Both can also be done from Telegram, later or from your phone: `/claude_login` and `/codex_login` in a private chat with the bot.

New chats start on Claude Code. On Codex only? Send `/version` and pick `v15_codex`.

### Everyday commands

```bash
./augctl status          # containers and health
./augctl logs            # follow the app's log
./augctl update          # pull, rebuild, restart
./augctl down            # stop (your data stays)
./augctl secret NAME     # store a key the agent can use but never read
./augctl shell           # a shell inside the app's container
```

`AUG_PORT` changes the API port (8012 by default). `COMPOSE_PROJECT_NAME` runs a second copy beside the first. Set them when you first run `setup`: it saves both in `.env`, where later commands find them.

Running the installer again updates the code and keeps your `.env`. It only offers to sign in again when nothing is signed in yet.

### Where things live

| Path | What |
|---|---|
| `data/workspace/AGENTS.md` | The assistant's persona and approach. Edit it freely |
| `data/workspace/memory/` | What it remembers, as Markdown |
| `data/workspace/skills/` | Its skills |
| `data/workspace/files/` | Files you sent and files it made |
| `data/claude/`, `data/codex/` | The agent CLIs' sessions, so chats resume after a restart |
| `data/.hushed/` | Secrets, encrypted (the key sits beside them: protect the folder) |
| Docker volume `aug_postgres` | Scheduled tasks, OAuth connections, the Kimi and GLM agents' chat history |

**Backups.** Copy `data/` and dump the database:

```bash
docker compose -f docker-compose.standalone.yml exec postgres pg_dump -U aug aug > aug.sql
```

### On a server

The same installer works on any Linux host with Docker. The API binds to `127.0.0.1` only. Telegram uses long polling, so the server needs no open ports and no domain.

You need a public HTTPS address only for two things: OAuth sign-ins to other services (Gmail, generic OAuth providers) from another device, and the Telegram mini app. Put a reverse proxy (Caddy, Cloudflare Tunnel, Tailscale Funnel) in front of port 8012 and set `BASE_URL` to that address.

## Manual setup

```bash
git clone https://github.com/vadimtitov/aug.git && cd aug
cp .env.example .env     # then edit it
./augctl up
./augctl login claude
```

The settings `docker-compose.standalone.yml` reads from `.env` are listed in [Configuration](configuration.md). The minimum:

```bash
API_KEY=<random string>            # REST API key
OAUTH_ENCRYPTION_KEY=<openssl rand -base64 32>
TELEGRAM_BOT_TOKEN=<from @BotFather>
TELEGRAM_ALLOWED_CHAT_IDS=<your user id>
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=                       # empty: no voice notes, no image generation
BASE_URL=http://localhost:8012
```

Don't know your user id? Leave `TELEGRAM_ALLOWED_CHAT_IDS` empty, start AUG and message the bot. Until the list is set, it lets nobody in and only replies with your id. Put it in `.env` and run `./augctl up`.

## Portainer and production

`docker-compose.prod.yml` is the fuller stack: an external Postgres, host folders under `/opt/aug`, the Telegram mini app (settings, skills, a live view of the agent's browser) and a Chromium the agent can drive through the Playwright MCP server.

1. Create the host folders, owned by UID 1000 (the app's user):
   ```bash
   sudo mkdir -p /opt/aug/{data,chrome-profile,browser-downloads}
   sudo chown -R 1000:1000 /opt/aug
   ```
2. In Portainer: **Stacks → Add stack → Repository**. Use this repository and the compose path `docker-compose.prod.yml`. Portainer builds from source, so no registry is needed.
3. Set the environment variables: at least `API_KEY`, `DATABASE_URL`, `LLM_BASE_URL`, `LLM_API_KEY`, `BASE_URL` (https), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_CHAT_IDS`, `OAUTH_ENCRYPTION_KEY` and `CHROME_PASSWORD`.
4. Deploy. Turn on GitOps updates to redeploy when `main` changes.
5. Send `/claude_login` or `/codex_login` to the bot.

The same file works with plain `docker compose -f docker-compose.prod.yml up -d` once those variables are in your environment.

**The mini app** needs `BASE_URL` and `MINIAPP_API_URL` on public HTTPS. Register its URL with @BotFather (**Bot Settings → Menu Button**).

## Development

`docker-compose.yml` is the development stack: source mounted with live reload, Postgres on 5432, the mini app's dev server on 5173. Telegram sign-in checks are bypassed, so never expose it.

```bash
cp .env.example .env
make run           # build, start, follow logs
make check         # lint, format check, tests
```

## When something's off

- **`./augctl logs`** first. Then `/status` in Telegram, which shows the model, the context and the plan's limits.
- **"The Claude login is missing, expired or revoked."** Run `./augctl login claude`, or send `/claude_login`.
- **Port already allocated.** Set `AUG_PORT` in `.env` and run `./augctl up`.

## Uninstalling

```bash
./augctl down
docker compose -f docker-compose.standalone.yml down -v   # also deletes the database
rm -rf ~/aug                                              # the code and data/
```
