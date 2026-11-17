# Security

AUG gives a capable agent a shell, your accounts and your network. This page says what holds it in and where those limits stop.

## The container is the boundary

Claude Code and Codex run inside AUG's container without approval prompts and without their own sandboxes. Inside it, the agent can run anything as an unprivileged user and write anywhere it has permission. It can also reach the network: the internet, and whatever the container can see on your LAN.

So the decision that matters is what you let the container reach. The standalone stack mounts only `./data`, publishes the API on `127.0.0.1` only, and keeps Postgres and the stealth browser off the host network entirely. Don't mount your home directory or the Docker socket into it.

## Who can talk to it

- **Telegram.** Only the user ids in `TELEGRAM_ALLOWED_CHAT_IDS` get through. The check is on the sender, so it holds in groups too. If the list is empty, nobody does: the bot only replies with the sender's id, and the mini app turns everyone away.
- **REST API.** Every route needs `API_KEY` or a mini app session. The exceptions are `/health`, the mini app's sign-in and the OAuth callbacks, and a callback does nothing without a valid single-use state.
- **The mini app** authenticates with Telegram's signed init data. `DEV_AUTH_BYPASS` turns that check off and is for development only.

## Secrets

- **Redaction.** Secrets live in [hushed](https://github.com/vadimtitov/hushed), a vault the agent can use but not read. Every shell command the agent runs, through Claude Code, Codex or AUG's own `run_bash`, goes through `hushed run`. The one exception is Claude Code's Monitor tool, which streams output line by line. It's checked against the blocked paths, and it runs without the secrets at all. Secrets arrive as environment variables, and their values are replaced with `[REDACTED]` in everything the command prints. The agent can use a key without it ever entering the model's context or the logs.
- **What redaction doesn't do.** It matches the literal value, so an agent set on revealing a key could encode it first. Nor does it stop the agent from *using* a key in a way you didn't intend, like sending it somewhere. It keeps secrets out of the transcript by accident; it is not a wall against a determined agent.
- **Clean environments.** The Claude Code and Codex processes start with an allowlisted environment. AUG's own keys (database URL, bot token, API key) are in hushed like any other secret: commands can use them, but their values stay redacted.
- **OAuth tokens** are encrypted at rest with `OAUTH_ENCRYPTION_KEY`. The agent calls APIs through a loopback gateway that attaches the token for it, so it uses a connection without reading the token.
- **Protect `data/`.** The hushed store and its key sit side by side in `data/.hushed/`, which only matters if someone gets the folder itself. Back it up as carefully as a password manager's vault.

## Guards

- **Blocked paths.** These are the secret store, SSH keys, OAuth and Gmail credentials, Codex's home and `settings.json`. They're refused in shell commands, file reads and writes, and outgoing files. This catches mistakes, not a determined agent: a command can reach a file without spelling out its path.
- **Your own blocklist.** `tools.bash.blacklist` in `settings.json` refuses commands matching your patterns, on every agent.
- **Claude Code's built-in tools** are an allowlist. Anything else is refused, including tools a future CLI version adds.
- **The Codex guard fails closed.** A tool-call shape it doesn't recognise, a crash or a timeout all count as a refusal. Codex's own default would let the call through.
- **SSH.** Every command on a remote host needs your approval in Telegram, unless you saved a rule that covers it ("always allow `df -h` on homeserver"). Host keys are pinned when the target is added.
- **Sign-in codes** pasted into Telegram (`/claude_login`) are deleted at once and never reach the agent.

## Prompt injection, and the worst case

The agent reads web pages, emails and files written by other people, and any of them can carry instructions. Claude and GPT resist this well, but not perfectly. If one got through, the agent could do whatever its connections allow:

- run commands in its container, and reach services on your LAN;
- read and send email on connected Gmail accounts;
- start, stop or redeploy containers, if you gave it a Portainer token;
- switch Home Assistant devices that carry the `aug` label.

On another machine over SSH, it can only ask: each command waits for your approval in Telegram. Keep the blast radius small. Connect only what you need, give Portainer the token of a user that sees only what AUG must manage, and save SSH approval rules narrowly.

## Reporting

Found a hole? Open a [private security advisory](https://github.com/vadimtitov/aug/security/advisories/new) rather than an issue.
