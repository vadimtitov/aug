# Context Compaction

## The problem

Every message, tool call, and tool response accumulates in the context window.
At 200K tokens the agent crashes mid-response and the user has to `/clear`.

---

## When it fires

- **Automatically** — when the thread hits **80% of the context window** (~160K tokens).
  Checked before every LLM call, including between tool calls mid-research.
- **Manually** — user sends `/compact`.

---

## The algorithm

```
                    Thread history
    ┌───────────────────────────────────────────────┐
    │  HumanMessage 1                               │
    │  [tool calls 1-8 + responses]                 │  ← "pre-run"
    │  AIMessage (answer)                           │
    │  HumanMessage 2                               │
    │  [tool calls 9-13 + responses]  ← current run │
    │  ...still going...                            │
    └───────────────────────────────────────────────┘
                         │
               80% threshold hit
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ STEP 1 — decide what to summarise (no LLM call yet)          │
    │                                                              │
    │ Count tokens in current run (tool calls 9-13).               │
    │                                                              │
    │ Can we free enough space by summarising pre-run only?        │
    │                                                              │
    │   YES → summarise pre-run only.                              │
    │          Keep current run tool calls as-is.                  │
    │                                                              │
    │   NO  → summarise everything (pre-run + current run tools).  │
    │          We have no choice — the current run is too heavy.   │
    └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ONE LLM call — cheap model, max_tokens=500                   │
    │                                                              │
    │ Input:  the messages selected in step 2                      │
    │ Output: a dense summary (≤500 tokens, forced by max_tokens)  │
    │                                                              │
    │ Drop the summarised messages from state.                     │
    │ Prepend the summary as a SystemMessage.                      │
    └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
              Agent continues with:
    ┌───────────────────────────────────────────────┐
    │  SystemMessage: "[Summary]: ..."  (~500 tok)  │
    │  [whatever was kept verbatim]                 │
    └───────────────────────────────────────────────┘
```

---

## "Pre-run" vs "current run"

- **Current run** = everything after the most recent HumanMessage.
  These are the tool calls the agent is actively working on right now.
- **Pre-run** = everything before that. Older turns, previous answers, history.

---

## What "enough space" means in Step 2

After compaction, the thread must fit comfortably under the trigger threshold.
Rule: if the current run alone (after step 1) is **less than 50% of the context
window** (~100K tokens), summarising pre-run only will free enough space.
If the current run is already bigger than that, we have no choice but to include it.

---

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `compaction_threshold` | `0.80` | Fraction of context window that triggers compaction |
| `context_window` | `200_000` | Token limit for this agent's model |
| `compaction_model` | `None` | Cheap model used for summarisation. If `None`, compaction is disabled. |
| `max_summary_tokens` | `500` | Max tokens the summary is allowed to use — passed as `max_tokens` to the summarisation LLM call |

---

## What the industry does

Trigger thresholds across production systems:

| System | Trigger | Notes |
|---|---|---|
| Claude Code (CLI) | ~83.5% | Configurable via env var |
| Codex CLI | 90% | Hard clamp |
| Goose | 80% | Configurable via env var |
| LangGraph Deep Agents | 85% | Dynamic per model |
| Hermes Agent | 50% | Aggressive outlier |
| LangMem | User-defined | No default |

Almost nobody specifies a "compact to X%" target — the summary just replaces
history and ends up whatever size `max_tokens` allows. LangMem defaults to
**256 tokens** for the summary budget. We use 500.

---

## Layer 2 — large tool output externalisation (separate, not yet built)

When a tool returns a huge response, write it to a temp file instead of putting
it in the message stream. The agent gets a short notice and can read the file
via `run_bash` on demand.

```
Instead of: ToolMessage(content="<15,000 chars of webpage>")
Agent sees:  ToolMessage(content="Saved to /tmp/fetch_a3f9.txt (15,203 chars).
                                   Preview: 'The European Space Agency...'
                                   Use run_bash to read or search it.")
```

This prevents bloat from ever entering the context in the first place.
Applies to: `fetch_page`, `run_bash`, `run_ssh`.

---

## Layer 3 — contextual extraction (future)

For tools like `fetch_page`, use a cheap LLM to extract only the relevant
portion of the response relative to the user's query — instead of returning
the full page or writing it to a file.

Independent of Layers 1 and 2. Design later.
