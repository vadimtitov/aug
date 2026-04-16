# Telegram Forum Topics Setup

This guide sets up a private Telegram supergroup with forum topics so you can maintain
multiple independent conversations with the bot — each topic is a separate, persistent thread.

## 1. Disable bot privacy mode

By default bots in groups only receive commands, not plain text. Disable this once in BotFather:

1. Open `@BotFather` → `/mybots` → select your bot
2. **Bot Settings → Group Privacy → Disable**

## 2. Create the supergroup

1. Tap the compose icon → **New Group**
2. Add your bot as the initial member
3. Give the group a name (e.g. "AUG")
4. After creation: **Group Settings → Edit → Topics → Enable**

## 3. Verify the bot is a regular member (not admin)

> **Note:** If you added the bot to a group before disabling privacy mode in step 1,
> remove it and re-add it. Telegram caches the privacy setting at join time.

## 4. Configure allowed chat IDs

Your personal Telegram user ID is already in `TELEGRAM_ALLOWED_CHAT_IDS` (get it from
`@userinfobot`). No changes needed — the bot checks the sender's user ID, so your messages
are allowed in any chat.

## 5. Pick an agent

The group chat has no agent configured yet. Send `/version` in any topic and select an agent
from the buttons. This only needs to be done once per group.

## Usage

- **New conversation** — tap **New Topic** in the group, give it a name, start chatting
- **Switch context** — tap a different topic; the bot resumes that thread exactly where you left it
- **Resume old thread** — tap the topic; full history is preserved
- **Reset a topic** — topics are persistent by design; create a new topic instead of clearing
- `/clear` is disabled in named topics (it still works in the General topic)
