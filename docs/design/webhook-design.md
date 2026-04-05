# Webhook / Event Ingestion — Design Document

## Status
Not implemented. Previous stub removed pending proper design.

---

## Problem

External systems (Home Assistant, scripts, cron jobs, IoT devices) generate events
that the agent should be aware of and optionally act on. The naive approach — fire
an agent per event and forward the response to the user — has fundamental flaws:

- **No back-and-forth.** If the agent wants to ask the user a clarifying question,
  there is no channel for the reply to return to the same context.
- **Tool use is broken.** If the agent uses a tool and needs to continue the
  conversation (e.g. browsing, ordering something), that continuation has nowhere
  to go.
- **Spam.** Without memory, every noisy event (motion sensor, low battery) becomes
  a notification.
- **No learning.** User says "stop sending me this" — there is nowhere to store that
  preference in a way that affects future events.

---

## Intended use cases

### 1. Intelligent filtering
Home Assistant fires 50 events a day. The agent decides which ones are worth
surfacing. The user can reply "don't send me motion alerts during the day" in their
normal Telegram thread, and the agent updates its own behaviour accordingly.
Over time the agent gets calibrated to the user's noise tolerance.

### 2. Cross-source situation awareness
The agent receives events from multiple sources and reasons across them.
Example: "battery low on feeder" + user is travelling → proactive message
"you're not home and the feeder needs charging". Not just forwarding — connecting dots.

### 3. Autonomous monitoring with conditional escalation
User configures once: "watch server logs for OOM errors; restart the container if
it happens more than 3 times in an hour; notify me only if something goes wrong."
Agent acts autonomously, notifies only on escalation. Human in the loop only when
needed.

### 4. Event accumulation and digests
Instead of spamming per-event, the agent accumulates events and delivers a morning
summary. It needs a tool to write events to storage and a tool to query them
(by time range, source, type).

---

## Key requirements

- **Persistent event log.** Events must be written to storage as they arrive.
  The agent must have a tool to query them (filter by source, time range, etc.).

- **Agent memory across events.** The agent must remember user preferences
  ("don't notify about X") that were expressed in any conversation — Telegram or
  otherwise. This means the same memory/notes system used in regular conversations
  must apply here.

- **No broken agentic loops.** The agent must not be able to start a multi-turn
  conversation from a webhook event, because there is no return channel. Tool use
  that is fully autonomous (no user input needed mid-task) is fine. Tool use that
  requires follow-up is not.

- **One-way initiation only.** Webhook events can trigger a notification to the
  user's active interface. The user's reply to that notification goes into their
  normal thread (Telegram), not back to the event context.

- **Configurable agent.** The agent used for event triage should be configurable
  but default to a lightweight, tool-restricted variant suitable for filtering
  decisions.

- **No per-source thread accumulation.** Storing per-source conversation history
  in the checkpointer does not serve any purpose and wastes storage. Event context
  should come from the event log tool, not from LangGraph message history.

---

## Open questions

- Where is the event log stored? Postgres table (simplest, consistent with reminders)?
- What is the query interface for the agent — a tool that takes filters (source, time
  range, limit)?
- How does the user express preferences that affect webhook behaviour? Via the normal
  Telegram conversation + note/memory tools? Or a dedicated config mechanism?
- Should digest scheduling be driven by the existing cron/reminder infrastructure?
- Rate limiting: should there be a minimum interval between notifications per source
  to prevent spam even if the agent decides to notify?
