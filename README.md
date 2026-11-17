<table>
  <tr>
    <td>

# AUG

A personal assistant you host yourself. You talk to it in Telegram. It runs on **Claude Code** or **Codex**, signed in with the Claude or ChatGPT plan you already have.

It remembers you between conversations, does real work in its own container, and keeps your keys out of the transcript.

  </td>
    <td><img src="https://github.com/user-attachments/assets/2cf5088d-433e-4ca5-b6e1-3f6fc1909b00" width="220"></td>
  </tr>
</table>

```bash
curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
```

Needs Docker. Works on a Mac, a Linux box or a home server. Nothing listens on the network: Telegram needs no open port. [Other ways to install →](docs/installation.md)

## Why AUG

Claude Code and Codex are excellent agents, and they run on a plan you may already pay for. But they live in a terminal: they forget you when the session ends, and they can't message you first.

AUG keeps those agents and gives them what an assistant needs:

- **One conversation everywhere.** Telegram on your phone, with voice notes, photos, files and locations. Forum topics give you parallel threads. A REST API is there for everything else.
- **Memory.** It writes notes as you talk and folds them into plain Markdown files every night. They're yours to read and edit.
- **Time.** It sets reminders, runs recurring tasks and messages you first when something's done.
- **Hands.** It has a shell, Python and common CLI tools in its own container, web search and a page reader that gets past bot protection. It connects to Gmail, SSH hosts, Home Assistant, Portainer and any MCP server.
- **Keys out of the transcript.** Secrets sit in [hushed](https://github.com/vadimtitov/hushed). The agent can use them in commands, and their values are redacted from everything it reads back.

## The agents

Each chat runs on an agent you choose with `/version`, and you can switch at any time.

| Agent | Runs on | Signs in with |
|---|---|---|
| `v14_claude_code` (default) | Claude Code · Claude Opus 5.5 | your Claude plan (`/claude_login`) |
| `v14_codex` | Codex · GPT-6.1 Sol | your ChatGPT plan (`/codex_login`) |
| `v13_kimi`, `v13_glm` | AUG's own loop · Kimi K3, GLM-5.3 | an OpenAI-compatible endpoint you provide |

The `v15_*` versions are the same agents with a Home Assistant reflex: they switch your lights the moment you ask, before the agent answers.

Claude Code and Codex keep their own strengths: tool use, subagents, context compaction, skills and web search. AUG gives them their tools, the memory, the schedule and the chat, and keeps the parts that would break a personal assistant switched off. [How the agents are run →](docs/agents.md)

## Compared with

- **Claude Code or Codex on a server.** You keep the same agent. AUG adds a phone interface, sessions that survive restarts, memory across conversations, scheduled work and secret redaction.
- **OpenClaw and other gateways.** Those connect many chat apps to many models, mostly through API keys. AUG does one thing: it runs the vendors' own agents on your subscription, in a sandboxed container, behind Telegram.
- **ChatGPT and Claude apps.** Their memory and tools live in the vendors' clouds. AUG runs on your machine: it can reach your network, and its memory is files you own.

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Laptop, server, Portainer; updating and backups |
| [Agents](docs/agents.md) | Claude Code, Codex and the rest: sign-in, models, what's on and off |
| [Using AUG](docs/usage.md) | Telegram commands, memory, skills, tasks, tools, the REST API |
| [Configuration](docs/configuration.md) | Every setting |
| [Security](docs/security.md) | What the agent can and cannot do, and where the limits are |

## Notes

AUG is one person's daily assistant, published as it grows. It runs the official Claude Code and Codex CLIs for one person: you. How that use counts against your plan is up to Anthropic's and OpenAI's terms. Read yours.

MIT licensed.
