<table>
  <tr>
    <td>

# AUG

Your own assistant, built on Claude Code or Codex, with your current Claude or ChatGPT plan.

You talk to it in Telegram. It remembers you, keeps your schedule, writes to you when something needs you, and gets work done on a Linux machine of its own.

  </td>
    <td><img src="https://github.com/user-attachments/assets/2cf5088d-433e-4ca5-b6e1-3f6fc1909b00" width="220"></td>
  </tr>
</table>

```bash
curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
```

Runs in Docker on a Mac, a Linux box or a home server. Telegram needs no open ports, so nothing is exposed to the internet. [Read the script first](install.sh), or see [other ways to install](docs/installation.md).

## Why

Claude Code and Codex are the best agent harnesses there are, and a Claude or ChatGPT plan already pays for them. But they're coding agents. Remote Control and Codex cloud put a session on your phone, and it's still a coding session: tied to a repository, and blind to what a phone is good for, like a voice note or where you are.

AUG runs the same harnesses as a personal assistant:

- **One conversation everywhere.** Telegram on your phone: text, voice notes, photos, files, locations. Forum topics give you parallel threads, and a REST API covers everything else.
- **Memory.** It takes notes as you talk and consolidates them every night into plain Markdown files. The agent reads them in every conversation, and you can open and edit them too.
- **Time.** It sets reminders, runs recurring tasks and messages you first when something's done.
- **Its own machine.** A shell, Python and the usual tools, inside a container where it works without stopping to ask.

## What it can do

| | |
|---|---|
| **Chat** | Telegram: text, voice notes, photos, files, live location, forum topics as parallel threads. The REST API for anything else. |
| **Remember** | Notes during the conversation, folded nightly into Markdown memory files: who you are, what's going on, what it's learned. |
| **Schedule** | Reminders, recurring tasks, work it starts on its own and reports back on. |
| **Work** | A Linux shell with Python and common CLI tools. Files go both ways: send it a PDF, get a spreadsheet back. |
| **Browse** | Web search, and a page reader that gets past bot protection. |
| **Email** | Gmail across several accounts: search, read, draft, send. |
| **Reach your machines** | SSH to your servers, with your approval for every command. Docker through Portainer. |
| **Run your home** | Home Assistant: switches the devices you've labelled the moment you ask. |
| **Grow** | Skills you or the agent write, and any MCP server, installed by asking. |
| **Keep secrets** | API keys it can use in a command but never read. The values are masked everywhere. |

## The agents

Each chat picks its agent with `/version`, and you can switch at any time. Memory, skills, files and secrets are shared.

| Harness | Agents | Model | Runs on |
|---|---|---|---|
| Claude Code | `v15_claude_code` (default), `v14_claude_code` | Claude Opus 5.5 | your Claude plan |
| Codex | `v15_codex`, `v14_codex` | GPT-6.1 Sol | your ChatGPT plan |
| LangGraph (AUG's own) | `v15_kimi`, `v15_glm`, `v13_kimi`, `v13_glm` | Kimi K3, GLM-5.3 | per token, through any OpenAI-compatible endpoint |

The `v15` agents add the Home Assistant reflex. Claude Code and Codex keep what makes them good: their tool use, subagents, context compaction and web search. AUG adds its own tools, the memory, the schedule and the chat. [How the agents are run →](docs/agents.md)

## What it costs

Nothing beyond your plan. AUG drives the official Claude Code and Codex, signed in as you, so there's no API bill for the main thing. `/status` shows where you stand on your plan's limits.

An OpenAI API key, if you add one, turns on voice notes and image generation. The Kimi and GLM agents bill per token. The rest needs nothing more.

Anthropic and OpenAI decide how their plans may be used, and their terms change. AUG is built for one person's own use. Read your plan's terms.

## Safe to leave running

Claude Code and Codex work on their own, which is the point of an assistant. What keeps that safe is where they run. They get a container of their own, with only its data folder mounted, and a fixed set of tools. Your keys are masked in everything the agent reads. Commands on your other machines wait for your approval in Telegram. [Security →](docs/security.md)

## Compared with

- **Claude Code Remote Control and Codex cloud.** These are for reaching a coding session from your phone. AUG is the assistant built around the same harness. It's one standing conversation that remembers you, keeps a schedule and writes first. It also takes voice notes, photos and locations. For work on a repository, use those.
- **OpenClaw.** Broader: many messengers, many models, a large skills ecosystem. That also makes it more to set up. It's a third-party harness, so on Claude it runs on an API key, billed per token. AUG runs the official harnesses and stays on your plan.
- **The ChatGPT and Claude apps.** Their memory and tools live in the vendor's cloud. AUG runs on your machine, reaches your network and your servers, and keeps its memory in files you own.

## Who it's for

One person who pays for Claude or ChatGPT and has somewhere to run Docker: a laptop to try it on, a home server to keep it on. It talks in Telegram today. Interfaces are pluggable, and WhatsApp or a native iOS app is next.

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Laptop, server, Portainer; updating and backups |
| [Agents](docs/agents.md) | Claude Code, Codex and LangGraph: sign-in, models, limits |
| [Using AUG](docs/usage.md) | Telegram commands, memory, skills, tasks, tools, the REST API |
| [Configuration](docs/configuration.md) | Every setting |
| [Security](docs/security.md) | What the agent can reach, and where the limits are |

MIT licensed.
