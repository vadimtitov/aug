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
Model configured via `IMAGE_GEN_MODEL` env var (default: dall-e-3).
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
