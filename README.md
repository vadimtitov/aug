<table>
  <tr>
    <td>

# AUG

A personal assistant you host yourself. You talk to it in Telegram. It runs on **Claude Code** or **Codex**, Anthropic's and OpenAI's own agents, signed in with your Claude or ChatGPT plan.

It remembers you between conversations, does real work in its own container, and keeps your keys out of the transcript.

  </td>
    <td><img src="https://github.com/user-attachments/assets/2cf5088d-433e-4ca5-b6e1-3f6fc1909b00" width="220"></td>
  </tr>
</table>

```bash
curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
```

Needs Docker and about 7 GB of disk. Works on a Mac, a Linux box or a home server. Nothing is exposed beyond localhost, because Telegram needs no open port. [Read the script first](install.sh), or see [other ways to install](docs/installation.md).

## Why AUG

Claude Code and Codex are excellent agents. But they live in a terminal: they forget you when the session ends, and they can't message you first.

AUG keeps those agents and gives them what an assistant needs:

- **One conversation everywhere.** Telegram on your phone: text, voice notes, photos, files, locations. Forum topics give you parallel threads, and a REST API covers everything else.
- **Memory.** It takes notes as you talk and consolidates them into plain Markdown files. They're yours to read and edit.
- **Time.** It sets reminders, runs recurring tasks and messages you first when something's done.
- **Hands.** It has a shell, Python and the usual CLI tools in its own container, web search, and a page reader that gets past bot protection. It connects to Gmail, SSH hosts, Home Assistant, Portainer and any MCP server (plug-in tool servers).
- **Keys out of the transcript.** Secrets live in [hushed](https://github.com/vadimtitov/hushed). The agent can use a key in a command; the value is masked in everything it reads back.

## The agents

Each chat runs on an agent you choose with `/version`, and you can switch at any time.

| Agent | Runs on | Signs in with |
|---|---|---|
| `v15_claude_code` (default) | Claude Code · Claude Opus 5.5 | your Claude plan (`/claude_login`) |
| `v15_codex` | Codex · GPT-6.1 Sol | your ChatGPT plan (`/codex_login`) |
| `v13_kimi`, `v13_glm` | AUG's own loop · Kimi K3, GLM-5.3 | an API endpoint you pay per token |

The `v15_*` versions carry a Home Assistant reflex: when it's set up, they switch the devices you've labelled for AUG the moment you ask, before the agent replies. The `v14_*` versions are the same agents without it.

Claude Code and Codex keep their own strengths: tool use, subagents, context compaction, skills and web search. AUG adds its tools, the memory, the schedule and the chat. It switches off the parts that don't belong in a personal assistant. [How the agents are run →](docs/agents.md)

## What it costs

- **Chat and work:** your Claude or ChatGPT plan, through its usage limits. `/status` shows where you stand.
- **Optional, pay per use:** voice notes and image generation need an OpenAI API key. The Kimi and GLM agents need an endpoint serving those models. The Home Assistant reflex needs one with OpenAI's Decisions API. Without these, the rest works. [Details](docs/configuration.md#the-openai-compatible-endpoint).

AUG runs the official, unmodified CLIs, logged in as you, for you alone. Anthropic and OpenAI decide how their plans may be used this way, and their terms change. Read yours, and don't share one install between people.

## What to know first

- **No approval prompts.** Claude Code and Codex run without prompts, inside a container with open network access, including your LAN. The container is the boundary, so mount nothing into it you wouldn't hand to the agent. SSH commands on other machines do ask first. [Security →](docs/security.md)
- **One person, Telegram only.** Each install is one person's assistant. Other chat apps aren't supported.

## Compared with

- **Claude Code or Codex over SSH.** Same agent. AUG adds the phone, memory across conversations, sessions that survive restarts, scheduled work and secret masking. It gives up the terminal UI and your own `~/.claude` setup. If a shell is all you want, SSH is simpler.
- **OpenClaw.** It connects many chat apps to many models and has a large skills ecosystem; it is the broader tool. AUG is narrower: it runs the vendors' own agents on your plan, behind Telegram, with the memory and scheduling built in.
- **The ChatGPT and Claude apps.** Their memory and tools live in the vendors' clouds. AUG runs on your machine: it can reach your network and your files, and its memory is files you own. It also costs you the setup.

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Laptop, server, Portainer; updating and backups |
| [Agents](docs/agents.md) | Claude Code, Codex and the rest: sign-in, models, what's on and off |
| [Using AUG](docs/usage.md) | Telegram commands, memory, skills, tasks, tools, the REST API |
| [Configuration](docs/configuration.md) | Every setting, and what needs a paid endpoint |
| [Security](docs/security.md) | What the agent can and cannot do, and where the limits are |

AUG is one person's daily assistant, published as it grows. MIT licensed.
