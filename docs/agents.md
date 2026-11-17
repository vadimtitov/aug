# Agents

Each chat runs one agent; `/version` switches it. Memory, skills, files and secrets are shared, so switching loses nothing.

| Agent | Harness | Model | Runs on |
|---|---|---|---|
| `v15_claude_code` (default) | Claude Code | Claude Opus 5.5 | your Claude plan |
| `v15_codex` | Codex | GPT-6.1 Sol | your ChatGPT plan |
| `v15_kimi`, `v15_glm` | LangGraph | Kimi K3, GLM-5.3 | per token |

`v15` agents add the [Home Assistant reflex](#home-assistant-reflex). The `v14` and `v13` agents are the same without it.

## Claude Code

The official CLI, through the Claude Agent SDK, pinned in the image.

- **Its own tools:** Bash, Read, Write, Edit, WebSearch, subagents, Monitor, Skill. No approval prompts: the container is the sandbox. Everything else is refused, including tools a future version adds.
- **Plus AUG's:** memory notes, files, tasks, Gmail, SSH, images, Portainer, MCP, and `web_fetch`, which gets past bot protection.
- **Each chat is a session** that survives restarts. It stays warm for an hour, as long as the prompt cache lives.
- **Sign-in:** a one-year token from `claude setup-token`. Renew with `/claude_login`.

## Codex

The official binary's app-server, pinned in the image.

- **Its own tools:** shell, patch editing, web search, image viewing, subagents. No approval prompts.
- **Plus AUG's,** the same set.
- **One server for all chats,** so the ChatGPT login is never rotated out from under another process. Threads survive restarts.
- **Sign-in:** ChatGPT's device code, `/codex_login`. It refreshes itself.
- A newly installed MCP server reaches a Codex chat after `/clear`.

## Kimi and GLM

AUG's own LangGraph loop, for open-weight models. AUG supplies everything: shell, search, page reading, compaction, subagents. Point `LLM_BASE_URL` at [Fireworks](https://fireworks.ai) or a [LiteLLM](https://github.com/BerriAI/litellm) proxy serving `kimi-k3`, `glm-5p3` and `glm-5p3-flash`. Billed per token.

## Background work

Background work asks for a tier and each harness picks the model: Opus or Haiku 5.5, GPT-6.1 Sol or GPT-6 Luna, Kimi K3 or GLM-5.3 Flash. The nightly memory pass runs on the default agent, so it needs only the Claude login.

## Limits

`/status` shows the model, how full the context is and your plan's five-hour and weekly windows. A turn that hits a limit tells you when it resets. Context compacts on its own; `/compact` forces it.

## Home Assistant reflex

A fast model reads each message and switches your devices before the agent replies: "lights off" is instant. Only entities labelled `aug` are in play. Needs `HASS_URL`, `HASS_TOKEN` and an endpoint serving `gpt-6-luna`. [Design](design/home-assistant-reflex.md)

[Harness design](design/harness-design.md) covers what all three share.
