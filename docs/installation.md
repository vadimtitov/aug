# Installation

You need Docker with Compose v2 (on a Mac: Docker Desktop, OrbStack or Colima), a Claude or ChatGPT plan, and a Telegram bot: message [@BotFather](https://t.me/BotFather), send `/newbot`, copy the token.

Why Docker? The agent works without asking, so it gets a machine of its own rather than your home directory.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
```

It clones AUG into `~/aug` (`AUG_DIR` to change) and walks you through setup: your bot token, a message to the bot so it learns your Telegram id, an optional OpenAI key for voice and images, then the sign-in. The first start downloads a few GB of prebuilt images.

Then say hello to your bot.

Run it again to update. Your `.env` stays.

## Sign in

```bash
./augctl login claude
./augctl login codex
```

**Claude:** paste a token from `claude setup-token`, or press Enter for a sign-in link. It lasts a year. **Codex:** open the link and enter the code. It refreshes itself.

Or from your phone: `/claude_login` and `/codex_login` in a private chat with the bot.

New chats start on Claude Code. Codex only? `/version` → `v15_codex`.

## Everyday

```bash
./augctl status     # containers and health
./augctl logs       # follow the log
./augctl update     # pull code and images, restart
./augctl down       # stop; your data stays
./augctl secret K   # a key the agent can use but never read
./augctl shell      # a shell in the container
```

## Your data

Everything is in `data/`, plus a Postgres volume for tasks and connections.

| | |
|---|---|
| `data/workspace/AGENTS.md` | The assistant's persona. Edit freely |
| `data/workspace/memory/` | What it remembers |
| `data/workspace/skills/` | Its skills |
| `data/workspace/files/` | Files in and out |
| `data/.hushed/` | Secrets, with their key beside them. Guard this folder |

Back up `data/` and the database:

```bash
docker compose -f docker-compose.standalone.yml exec postgres pg_dump -U aug aug > aug.sql
```

## On a server

Same installer. The API listens on `127.0.0.1` and Telegram uses long polling, so no ports and no domain.

A public HTTPS address is needed only for the Telegram mini app and for connecting Gmail from another device. Put Caddy, Cloudflare Tunnel or Tailscale Funnel in front of port 8012 and set `BASE_URL`.

## Manual setup

```bash
git clone https://github.com/vadimtitov/aug.git && cd aug
cp .env.example .env    # fill it in
./augctl up
./augctl login claude
```

Don't know your Telegram id? Leave `TELEGRAM_ALLOWED_CHAT_IDS` empty and message the bot: it lets nobody in and replies with your id. Settings: [Configuration](configuration.md).

## Portainer

`docker-compose.prod.yml` adds the mini app (settings, skills, a live view of the agent's browser) and a Chromium for the Playwright MCP server, with an external Postgres.

1. `sudo mkdir -p /opt/aug/{data,chrome-profile,browser-downloads} && sudo chown -R 1000:1000 /opt/aug`
2. **Stacks → Add stack → Repository**, compose path `docker-compose.prod.yml`.
3. Set `API_KEY`, `DATABASE_URL`, `LLM_BASE_URL`, `LLM_API_KEY`, `BASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_CHAT_IDS`, `OAUTH_ENCRYPTION_KEY`, `CHROME_PASSWORD`.
4. Deploy, then `/claude_login` in the bot.

The mini app needs `BASE_URL` and `MINIAPP_API_URL` on HTTPS, and its URL set in @BotFather (**Bot Settings → Menu Button**).

## Development

```bash
cp .env.example .env
make run      # live reload; never expose it, sign-in checks are off
make check    # lint and tests
```

## Trouble

- `./augctl logs`, then `/status` in Telegram.
- **"The Claude login is missing, expired or revoked":** `./augctl login claude`.
- **Port already allocated:** set `AUG_PORT` in `.env`, then `./augctl up`.

## Uninstall

```bash
./augctl down
docker compose -f docker-compose.standalone.yml down -v   # deletes the database
rm -rf ~/aug
```
