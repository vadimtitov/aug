# Security

AUG hands a capable agent a shell, your accounts and your network. Here's what holds it in, and where that stops.

## The container is the boundary

Claude Code and Codex run without approval prompts and without their own sandboxes. Inside the container the agent can run anything as an unprivileged user and reach the network, including your LAN.

So what matters is what the container can reach. The standalone stack mounts only `./data` and publishes the API on `127.0.0.1`. Never mount your home directory or the Docker socket.

## Who gets in

- **Telegram:** only the user ids in `TELEGRAM_ALLOWED_CHAT_IDS`, in groups too. Empty list, nobody: the bot only replies with the sender's id.
- **REST API:** `API_KEY` on every route except `/health` and the sign-in callbacks.
- **Mini app:** Telegram's signed init data.

## Secrets

- **Usable, never visible.** Keys live in [hushed](https://github.com/vadimtitov/hushed). Shell commands get them as `$NAME`, and their values are replaced with `[REDACTED]` in all output. Claude Code's Monitor tool runs without them.
- **Not a wall.** Redaction matches the literal value. A determined agent could encode a key, or use it somewhere you didn't intend.
- **OAuth tokens** are encrypted and attached by a local gateway, so the agent uses a connection without seeing the token.
- **Guard `data/.hushed/`.** The key sits beside the store.

## Guards

- Secret stores, SSH keys, credentials and `settings.json` are refused in commands, file access and outgoing files. This catches accidents, not a determined agent.
- `tools.bash.blacklist` in `settings.json` refuses your own patterns.
- Claude Code's built-in tools are an allowlist. The Codex guard fails closed.
- Every SSH command waits for your approval, unless a rule you saved covers it. Host keys are pinned.
- Sign-in codes pasted in Telegram are deleted at once and never reach the agent.

## Worst case

Web pages, emails and files can carry instructions. Claude and GPT resist this well, not perfectly. If one got through, the agent could do what its connections allow: run commands in its container, reach your LAN, use connected Gmail, manage containers through Portainer, switch `aug`-labelled devices.

Keep the blast radius small: connect only what you need, give Portainer a narrow token, keep SSH rules tight.

Found a hole? Open a [private security advisory](https://github.com/vadimtitov/aug/security/advisories/new).
