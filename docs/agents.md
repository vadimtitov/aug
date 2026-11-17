# Agents

An *agent* in AUG is a profile: a model, its tools and the engine that runs it. AUG calls that engine a *harness*. Every chat (a Telegram DM, a forum topic, a REST `context_id`) uses one agent. `/version` switches it; new chats start on `v15_claude_code`.

| Agent | Harness | Model | Sign-in |
|---|---|---|---|
| `v15_claude_code` | Claude Code | Claude Opus 5.5, high effort | Claude plan |
| `v15_codex` | Codex | GPT-6.1 Sol, medium effort | ChatGPT plan |
| `v13_kimi` | LangGraph (AUG's own loop) | Kimi K3 | OpenAI-compatible endpoint |
| `v13_glm` | LangGraph | GLM-5.3 | OpenAI-compatible endpoint |

The `v15_*` agents carry the [Home Assistant reflex](#home-assistant-reflex). `v14_claude_code` and `v14_codex` are the same agents without it; `v15_kimi` and `v15_glm` are Kimi and GLM with it.

Every interface talks to all three harnesses the same way: Telegram, the REST API, scheduled tasks and background jobs. Switching a chat's agent changes nothing else. Memory, skills, files and secrets are shared.

## Claude Code

AUG runs the official Claude Code CLI through the Claude Agent SDK. The version is pinned and ships inside the image.

- **Each chat is its own Claude Code session.** It runs as a warm process that closes after an hour idle, which is how long the prompt cache lives. The next message resumes it from Claude's session file, so a chat survives restarts and redeploys.
- **Claude keeps its own tools:** Bash, Read, Write, Edit, WebSearch, subagents (Task), Monitor, Skill and ToolSearch. They run without approval prompts, because the container is the sandbox. Every other built-in is refused, including any a future CLI adds. That covers Claude's own cron and wake-ups (AUG does the scheduling), remote triggers, messaging between sessions and WebFetch.
- **AUG adds its own tools** through an in-process MCP server: memory notes, files back to you, tasks, Gmail, SSH, image generation, Portainer, MCP management, and `web_fetch`. It reads pages like WebFetch but gets past bot protection. A few are always loaded; Claude finds the rest with ToolSearch.
- **The prompt** is Claude Code's own, plus AUG's: your `AGENTS.md`, the memory index, the chat's context. Nothing loads from `~/.claude` or project folders. Your skills load as a plugin; Claude's bundled skills are hidden.
- **Sign-in** is a one-year token from `claude setup-token`, kept in hushed. It never refreshes, so it can't fail on a refresh. Renew it with `/claude_login` or `./augctl login claude`.

## Codex

AUG drives the official Codex binary's app-server over JSON-RPC. The binary is pinned and ships inside the image.

- **One Codex server for every chat.** A ChatGPT login rotates its refresh token, so two processes would sign each other out. Each chat is a Codex thread on that server, resumed from Codex's own files after a restart.
- **Codex keeps** its shell, patch editing, web search, image viewing and subagents, without approval prompts or Codex's sandbox: again, the container is the boundary. Off: Codex's own memory, goals, apps, plugins and image generation.
- **AUG's tools** are the same set as on Claude Code, given as Codex dynamic tools.
- **Sign-in** is ChatGPT's device code: `/codex_login` or `./augctl login codex`. Codex refreshes the login, and AUG refreshes it daily besides, so it never lapses unused.
- A Codex thread keeps the tools it started with. A newly installed MCP server reaches a chat after `/clear`.

## Kimi and GLM

AUG's original agents run on its own LangGraph loop with open-weight models. Here AUG provides everything: the shell (`run_bash`), Brave web search, page fetching, compaction and subagents.

They need an OpenAI-compatible endpoint that serves `accounts/fireworks/models/kimi-k3`, `glm-5p3` and `glm-5p3-flash`. That can be [Fireworks](https://fireworks.ai) directly (`LLM_BASE_URL=https://api.fireworks.ai/inference/v1`), or a [LiteLLM](https://github.com/BerriAI/litellm) proxy when you also want OpenAI's models on the same endpoint. You pay per token.

## Model tiers

Background work asks for a tier, not a model, and each harness picks its own:

| Tier | Claude Code | Codex | Kimi / GLM |
|---|---|---|---|
| Strong | Claude Opus 5.5 | GPT-6.1 Sol | Kimi K3 / GLM-5.3 |
| Fast | Claude Haiku 5.5 | GPT-6 Luna | GLM-5.3 Flash |

`web_fetch` reads pages with the fast tier of the agent that called it.

**Memory consolidation** (the nightly and weekly "dream") runs on the default agent, `v15_claude_code`, so it needs only the Claude login.

## Usage and limits

`/status` shows the chat's model, how full its context is, and your plan's five-hour and weekly windows. A turn that hits a limit ends with the time it resets. Claude Code and Codex compact their context themselves, and `/compact` does it on demand.

On Claude, AUG's use counts as Agent SDK use. It draws on the monthly Agent SDK credit that comes with your plan, not on the limits of interactive Claude Code, and pauses when the credit runs out unless you've turned on extra usage. On ChatGPT, Codex counts against your plan's Codex limits.

Each AUG install is one person's assistant, running the vendors' own CLIs with that person's login. Anthropic's and OpenAI's terms decide how a plan may be used, and they change. Check yours.

## Home Assistant reflex

The `v15_*` agents switch your Home Assistant devices before the agent replies. A small, fast decision model reads each message and acts at once when the intent is clear ("lights off"). The agent's turn then goes on as usual. Only entities with the `aug` label are in play.

It needs `HASS_URL`, `HASS_TOKEN`, and an `LLM_BASE_URL` that serves OpenAI's Decisions API model. [Design and evaluation](design/home-assistant-reflex.md).

## Under the hood

[Harness design](design/harness-design.md) covers the contract all three implement: turns, steering, `/stop`, compaction, unsolicited turns, sessions and their records, and the guards.
