# Tool Ideas

## Reminders ✅
Trigger a message to the user at a future time.
DB-backed (PostgreSQL reminders table), polled every 30s in a background task.
Survives server restarts. Tool: `set_reminder(when, message)` — when is ISO 8601.
Requires `TELEGRAM_DEFAULT_CHAT_ID` to be set.

## Vision (image input) ✅
Let the user send images and have the agent read/analyze them.
Uses the same model as the agent — no separate config needed.
Claude and GPT-4o are already multimodal.
Telegram photos and stickers are both handled.

## Image generation ✅
Generate images from text prompts.
Tool: `generate_image(prompt)` — uses LiteLLM proxy.
Model configured via `IMAGE_GEN_MODEL` env var (default: gpt-image-1.5).
Returns image as attachment sent directly to the user.

## Portainer ✅
Manage containers, stacks, and deployments via Portainer's REST API.
Tools: `portainer_list_containers`, `portainer_container_logs`,
`portainer_restart_container`, `portainer_list_stacks`.
No socket mount needed — just `PORTAINER_URL` + `PORTAINER_API_TOKEN`.
`PORTAINER_ENDPOINT_ID` defaults to 1.

## Webhook endpoint ✅ (not a tool)
`POST /webhook/event` — external services send events here.
Agent decides whether to notify the user via Telegram.
Responds with `<noanswer>` to stay silent, otherwise sends the message.
Auth: same X-API-Key header. Requires `TELEGRAM_DEFAULT_CHAT_ID`.

## Twilio SMS ✅
Send a real SMS to any phone number.
Tool: `send_sms(to, message)`.
Requires `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`.
Note: implemented in `aug/core/tools/twilio_sms.py` but not yet registered in any agent.

---

## Phone Call
The agent calls a number, has a real two-way conversation to accomplish a goal, and
reports back with a transcript and summary. "Call the dentist and find out their earliest
slot next week." User never touches the call — fully delegated.

Architecture: dedicated voice subagent (same pattern as browser tool), not part of the
main LangGraph. The `phone_call` tool kicks off the call and returns immediately; result
arrives via Telegram when the call ends. The voice pipeline runs independently:
STT (streaming) → LLM → TTS, ~885ms mouth-to-ear latency — acceptable since the user
isn't waiting on hold, the agent is.

Three implementation paths (pick one):
- **Pine AI** — managed, zero infra, OpenClaw uses this. `POST /call` with number +
  objective, SSE for result. ~$0.30–0.50/call. English only, Pro subscription required.
- **Vapi** — flexible managed service, supports custom LLM (point at LiteLLM proxy),
  custom voices, tool calling mid-call. Webhook delivers transcript. ~$0.13/min.
- **Twilio ConversationRelay + LiteLLM** — self-hosted. Twilio has a tutorial using
  FastAPI + LiteLLM specifically — AUG's exact stack. ~$0.05–0.10/min in API costs.

Tool: `phone_call(to, goal)`.
References: github.com/19PINE-AI/openclaw-pine-voice, docs.vapi.ai/calls/outbound-calling,
twilio.com/en-us/products/conversational-ai/conversationrelay

## Recurring Tasks
Cron-style scheduling for the agent itself. Unlike `set_reminder` (one-shot), this lets
the agent run a task on a repeating schedule: daily briefings, weekly reviews, periodic
health checks, habit pings.

Tool: `schedule_task(cron, task)` — cron is a standard cron expression, task is a
prompt the agent will run at each firing. DB-backed, survives restarts.
Requires `TELEGRAM_DEFAULT_CHAT_ID` to deliver results.

## Text-to-Speech (TTS)
Generate an audio file from text and send it as a Telegram voice message. The agent
responds with audio when explicitly asked ("read this to me", "respond with voice").
No automatic inference — user requests it; text remains the default.

Tool: `text_to_speech(text)`. Uses OpenAI TTS API via LiteLLM proxy.
Cost: ~$15/million chars (standard quality) — negligible at personal use scale (~$0.01
per typical response).

## Spawn Subagent
Let the main agent spawn a fully-equipped AugAgent instance with a task, run it to
completion, and return the result. The subagent has full tool access.

Unlocks parallel execution ("research these 5 things simultaneously") and task
delegation ("handle this in the background while we talk"). Makes AUG an orchestrator,
not just a single thread.

Tool: `spawn_subagent(task, tools?)` — runs another agent turn, returns its final output.
Implementation: reuse the existing agent invocation path with a fresh thread.

## Telegram User API (Telethon)
Read arbitrary Telegram channels, groups, and chats that the agent's user account is a
member of — things the Bot API can never access. Uses Telethon (MTProto), which is
documented and widely used; Telegram tolerates personal automation.

Use cases: monitor a border-crossing queue chat, track a local community group, search
across channels for information that exists nowhere else.

Before building: test whether `browser` + `t.me/s/<channel>` is sufficient for public
channels. If yes → pure skill, no code needed. If not → Telethon route:
- Add Telethon to Dockerfile
- One-time interactive phone auth (outside the agent), session file persists
- `run_bash` + skill handles all subsequent operations

## Google Calendar
Read and write Google Calendar. Natural companion to Gmail — the OAuth infrastructure
already exists.

Tool: `calendar(action, ...)` — actions: `list_events`, `create_event`, `update_event`,
`delete_event`, `find_free_slots`.
Unlocks full scheduling assistant: "schedule a meeting next week when we're both free",
"what's on my agenda today?", "block off Friday afternoon."

## Google Contacts
Search and manage Google Contacts via the People API. Complements Gmail and Calendar.
Useful for phone calls ("call mom") and anywhere a name needs to resolve to a number or
email.

Tool: `contacts(action, ...)` — actions: `search`, `get`, `create`, `update`.
Not required for phone calls (can pass numbers directly) but makes them much more natural.
