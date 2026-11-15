# Home Assistant reflex

The `v15_*` profiles switch smart-home devices the moment the user asks, before the agent
starts its turn. Example requests: "turn off the hallway light", "make the lamp red and the
sofa light 20%". This document covers how the reflex decides and how it was evaluated
before shipping.

Code: `aug/core/reflexes/homeassistant.py`, `aug/utils/decisions.py`, `Harness.run()` in
`aug/core/harness/base.py`.

## Why it was rebuilt

The previous reflex asked a fast chat model, with function calling, to return Home
Assistant service calls for every message. It acted far too often. Devices switched on
during conversations that had nothing to do with the home, for example a question about
the weather. No profile ran it.

Decision models answer a fixed question with a probability instead of generating text.
This gives a threshold to tune, and makes "do nothing" the default. The reflex uses OpenAI's
Decisions API (`gpt-6-luna`), reached through the LiteLLM proxy's OpenAI pass-through.

## How it works

1. **Layer 1, every message.** One yes/no question: is the user's last message an
   instruction to operate a home device right now? The prompt is `HA_REFLEX_GATE`. Below
   0.7 the reflex stops, and nothing else is asked or done.
2. **Layer 2, only after a yes.** One request asks, for every device labelled for AUG in
   Home Assistant:
   - is this device targeted?
   - what should happen: on, off, open, close, position, heat, temperature, play, pause,
     volume?
   - which brightness, colour, white tone, position or temperature?

   Every value list has "not specified" and "other" options. The model is given the
   device list as context, including any aliases set in Home Assistant.
3. **Code turns the answers into service calls** (`plan()`). The reflex acts only when
   every answer is clear. It does nothing when:
   - a device's targeting falls between 0.3 and 0.7;
   - an answer has low confidence, or is refused or missing;
   - the answer is "other";
   - two answers contradict each other;
   - the device can't do what was asked (colour on a dimmer-only light, for example). The
     reflex never carries out half of a request.

   Doing nothing is always safe, because the agent then handles the message as usual.

**Devices the reflex will operate:**

- lights and switches;
- curtain-like covers: curtain, blind, shade, shutter or awning. It never operates a garage
  door, a gate, or a cover of unknown class;
- thermostats that can heat;
- media players.

**Where it runs.** `Harness.run()` runs the reflex before the turn, on turns someone is
there for. It skips pushes, scheduled tasks, background turns and follow-ups. What the
reflex did, including any failed call, is appended to the user's message as a note, so the
agent knows the action is done and never repeats it. The user sees a short confirmation
before the agent replies.

**Context.** The reflex sees the message and the two previous messages of the session,
each cut to 300 characters. They come from AUG's transcript, and messages from turns
nobody was there for are left out. This is what makes follow-ups like "now turn it on"
work.

**Timing.** Deciding (both layers) has 5 s (`DECIDE_SECONDS`). Once decided, all service
calls are sent at the same time and always reported. The whole reflex is capped at 12 s.

## Evaluation

### Data

- **Real messages:** 992 messages the owner sent to AUG, taken read-only from
  production. They include all 223 short messages containing device-related words
  ("light", "turn on", "curtains", "TV" and so on, in Russian and English), plus a random
  sample of the rest. 378 are transcribed voice messages, sent exactly as production sends
  them. Each message carries its two real preceding messages.
- **Synthetic messages:** 22 phrases covering two things the history lacks:
  - multi-device requests ("make the lamp red and the light 20%");
  - look-alikes that must not act ("I'll switch the kitchen light off myself").
- **Devices:** a snapshot of the 15 labelled devices.
- **No Home Assistant calls.** The evaluation only asked the model. Service calls were
  computed by the same `plan()` code and compared with the labels, never sent.

**Labels.** Every message was hand-labelled with what the reflex should do:

- a set of service calls;
- nothing;
- or "either is acceptable", for genuinely ambiguous requests such as "close the curtains"
  in a home with two sets of curtains.

| Real messages | Count |
|---|---:|
| Not a device request | 928 |
| Clear device request | 37 |
| Ambiguous device request (acting or not both acceptable) | 12 |
| Device request for something the reflex doesn't control (sleep mode, vacuum…) | 15 |

### Layer 1 prompt

Three gate prompts were compared on 300 labelled messages plus the synthetic set, at a 0.7
threshold. The chosen prompt names who acts and when, then lists what is not an
instruction. Its exclusions:

- the user narrating their own actions;
- discussing an approach;
- editing automations or scenes;
- prohibitions;
- questions;
- later times;
- figurative language.

| Gate prompt | Requests caught | False alarms |
|---|---:|---:|
| First draft (generic "is this a command") | 46/46 | 3/253 |
| **Shipped (`HA_REFLEX_GATE`)** | **46/46** | **0/253** |
| "Would the user expect this to happen now" framing | 46/46 | 4/253 |

### History size

Layer 1 on the 1,014 labelled messages, run through the shipped code with production-style
text:

| Earlier messages shown | Requests caught | False alarms |
|---|---:|---:|
| 0 | 43/46 | 3/940 |
| **2 (shipped)** | **46/46** | **1/940** |
| 4 | 46/46 | 2/940 |

Without history, follow-ups ("now turn them all on") are missed. Messages written while
designing an automation also look like commands. With more history, older context starts
to pull the model towards acting. The one false alarm at 2 is a borderline case: "not
approved yet, make it brighter", written while tuning a scene.

### End to end (shipped code)

| Real messages | Outcome |
|---|---|
| 928 not a device request | **0 actions.** Layer 1 let 1 through, and layer 2 found nothing to do. |
| 37 clear device requests | **32 done correctly**, 3 left to the agent (unsure which device), 2 also switched a neighbouring device |
| 12 ambiguous requests | left to the agent |
| 15 requests for devices outside the reflex | 0 actions |

Both wrong actions come from naming. "The lamp in the living room" also matched the sofa
light, whose Home Assistant name is "Living Room Dimmer Living Room Light". Aliases in
Home Assistant fix this without code changes.

The synthetic set gave the following results:

- **12 look-alikes:** no action.
- **9 requests:** 6 done correctly. 3 were left to the agent, including the red-lamp/20%
  example, because the TV backlight's targeting fell in the unsure band. One of those 3
  asked for a relative white tone ("warmer"), which the reflex leaves to the agent by
  design.
- **"Turn everything off in the living room"** also closed the curtains. Whether
  "everything" includes the curtains is arguable; it was counted as wrong.

### Latency and cost

| | p50 | p95 | Cost |
|---|---:|---:|---:|
| Layer 1, every message | ~150 ms | ~250 ms | ~$0.00005 |
| Layer 2, device requests only | ~300 ms | ~700 ms | ~$0.001 |

These are single sequential calls. Under the evaluation's six parallel calls they were
roughly double. The first call after a quiet spell can take up to about 2.5 s. The API
occasionally answers `503 Decision admission timed out`, and the reflex then does nothing.

## Known limitations

- **Naming.** Ambiguous device names cause both wrong actions and most of the "unsure"
  silences. Give devices Home Assistant aliases that match how they are spoken about.
- **Added latency.** A v15 turn waits for layer 1, about 150 ms, before the agent starts.
  `/stop` in that window finds no turn to stop.
- **Relative white tone** ("warmer", "cooler") is left to the agent.
- **Small labelled sample.** One person labelled the data, and the set has 37 clear real
  requests. Treat the request-side numbers as indicative. The false-alarm side (928
  messages, 223 of them device-related) is the stronger result.
- **Not in the repository.** The evaluation scripts and dataset stay out of this public
  repository, because the dataset is private chat history.
