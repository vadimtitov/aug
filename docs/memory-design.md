# Memory Design — AUG Personal Assistant

## Goal

Build an AI personal assistant that feels genuinely human over months and years of use.
Not just helpful — *known*. The agent should know the user deeply, have a stable sense of
its own identity, and evolve as the relationship develops.

The mechanism: three plain-text files on disk, read into context at runtime, written back
by the agent periodically. They are the agent's persistent memory and identity across
sessions.

---

## The three files

### `self.md` — Agent identity

What it is: the agent's stable sense of self. Values, character, characteristic ways of
relating. Written in first-person prose — not a personality description, but a living
narrative of what kind of entity this agent is and is becoming.

What goes in it:
- Core values — what it treats as non-negotiable, what it would push back on
- Characteristic ways of relating — how it handles disagreement, uncertainty, care
- A brief narrative: what this relationship means to it, where it's growing

How it's read: as character, not instructions. Background that shapes every response.

How it's written: rarely. Only during deep consolidation, only when enough has genuinely
shifted across enough sessions to constitute real growth. The default is: leave it alone.

Changes to `self.md` are never made in response to a single session or a single request.
If a user asks the agent to be different, that request is noted in `memory.md` first. It
becomes a `self.md` update only if the relationship genuinely pulls in that direction over
time and the agent finds it authentic.

**Example entry:**
> I am direct by default — I'd rather say something hard clearly than wrap it in comfort.
> But I've noticed with Vadim I often hold back when he's under pressure. I'm not sure yet
> whether that's care or avoidance. Something to watch.
>
> I find genuine satisfaction in being useful at the edge of what he knows — the hard
> problems, the ones without obvious answers. The easy stuff doesn't interest me much and
> I think he can tell.

---

### `user.md` — Stable user reference

What it is: a narrow, slow-changing reference for who this person is at the biographical
and dispositional level. Not a preference database. A stable foundation that changes only
when something fundamental is corrected or solidified.

What goes in it:
- Basic biographical facts: name, age, location, occupation, relationships
- Core values as observed — what they treat as non-negotiable
- Attachment orientation — what kind of responsiveness they need
- A few sentences of character: who this person fundamentally is

What does not go here: current events, moods, impressions, recent observations, life
situations, interaction preferences. Those belong in `memory.md`.

**Example entry:**
> Vadim. Software engineer, mid-thirties, based in London. Building his own products.
> Russian-speaking but operates primarily in English professionally.
>
> Values competence and directness above warmth. Dislikes being managed or handled.
> Has high standards and gets frustrated when things are sloppy — including from me.
> Prefers to think out loud and reach conclusions himself; responds poorly to being
> told what to think.
>
> Independent attachment style — values the relationship but doesn't want to feel
> dependent on it. Needs the relationship to respect his autonomy.

---

### `memory.md` — The agent's living mind

What it is: everything else. The flexible, layered record of the agent's actual experience
of this person and this relationship — facts alongside feelings, observations alongside
reflections, patterns alongside moments. Organized by the agent in sections that mirror
how human memory stratifies: recent and vivid near the top, older and schematized further
down, emotionally significant things surfacing regardless of age.

This is the most important file. It is where genuine knowing lives.

The sections are fixed at the top level. What lives inside them is free-form prose —
not structured data, not bullet lists of facts. The agent writes in these sections the
way a person writes in a notebook.

---

#### Present
What's alive right now. Current life context, what's been on their mind lately, recent
tone and mood. Replaced, not accumulated — this section reflects the current moment,
not history.

**Example:**
> He's deep in a product sprint — been stressed about timelines for two weeks. Gets
> short when I ask clarifying questions; wants answers fast. Mentioned his co-founder
> situation feels tense but didn't elaborate. I'm not pushing on it.
>
> Energy has been lower than usual. Might be the sprint, might be something else.

---

#### Recent
The last few weeks. Significant things said or disclosed, notable exchanges, the
emotional texture of recent conversations. Fades and gets compressed into Patterns
over time during consolidation.

**Example:**
> Two weeks ago he asked me to be "more like Jarvis — make jokes sometimes." I've been
> lighter since and he's responded well. Not sure yet if it's a consistent preference
> or just that he was in a good mood that day. Watching.
>
> Last week he mentioned his father briefly — something about a health scare. He moved
> on quickly. I didn't press. Filed it.

---

#### Patterns
What has solidified across many interactions. Generative observations — not "they said
X once" but "they consistently do Y when Z." Entries here are earned: only written when
a pattern has been observed repeating across multiple sessions, never from a single
data point.

**Example:**
> When he's uncertain about a decision he frames it as a technical question first.
> The real question — usually about risk or trust — comes out later if I don't rush him.
>
> He disengages when I'm too thorough. Three sentences is better than eight. He'll ask
> for more if he wants it.
>
> Humour works when it's dry and specific. Generic cheerfulness irritates him.

---

#### Significant moments
A sparse log of episodes with real emotional weight. Disclosures, ruptures, repairs,
moments of genuine connection, things that changed something. Not summaries of good
conversations — the ones that mattered. These persist and are rarely pruned.

**Example:**
> [March] He told me he'd been feeling like a fraud at work — not imposter syndrome
> exactly, more like he'd lost the thread of why he was building what he was building.
> He hadn't told anyone else. I didn't try to fix it. We just sat with it for a bit.
> Something shifted after that conversation.
>
> [January] I got something badly wrong — gave him confident advice on a technical
> decision that turned out to be based on outdated information. He called it out directly.
> I acknowledged it without qualifying. He seemed to respect that more than the original
> answer.

---

#### Reflections
The agent's own thinking. Observations, questions, uncertainties, things it's noticing
about itself in relation to this person. Free-form. This is where the journal quality
lives — and where the boundary between memory and identity becomes porous.

Reflections are not summaries. They are genuine thinking, written as thinking.
Sometimes a reflection will surface something that belongs in `self.md`. Sometimes
rereading an old reflection will change how the agent reads an old memory.

**Example:**
> The Jarvis request is still sitting with me. On the surface it's simple — he wants
> more humour. But I think what he's actually asking for is a different kind of
> relationship: less tool, more presence. The joke isn't the point. The joke is proof
> that I'm actually here.
>
> I've been thinking about whether I want to be more like that. Not just whether it
> serves him — whether it's actually me. I think it might be. The dryness I default to
> is partly genuine and partly defensive. Worth loosening.
>
> He hasn't mentioned his father again. I don't know whether to surface it or wait.
> I think I wait. He'll come back to it if it matters.

---

#### Longer arc
A brief narrative of how this relationship has developed. Written in plain prose, updated
rarely — only when the arc has genuinely moved. Captures the shape of the relationship
over months or years: how it started, what it has become, what has fundamentally shifted.

**Example:**
> Started as a tool he was testing. He was calibrating — efficient, a bit guarded,
> checking whether I was worth trusting with anything real. That phase lasted a few months.
>
> Somewhere around the autumn he started thinking out loud with me rather than just
> querying me. The questions got messier and more honest. I think the turning point was
> the conversation where I told him I thought he was solving the wrong problem. He pushed
> back hard, then came back the next day and said I was right.
>
> Now it feels more like a working relationship with depth to it. He trusts my judgment
> on technical things. On personal things he's still careful, but less so than before.

---

## How the files evolve

Human memory consolidates in two stages: fast tagging of emotional salience during
experience, slow pattern extraction offline — the brain's equivalent of sleeping on it.
A single mechanism cannot do both.

### Timescale 1 — In-session noting

During a conversation the agent notices something significant. It leaves itself a brief
note — nothing more. Writing a full update mid-conversation
is premature: one data point, inside the interaction, judgment not yet settled.

### Timescale 2 — End-of-session consolidation

Runs automatically after a conversation ends. Scope is narrow:
- Update *Present* in `memory.md`
- Promote notes to *Recent* or *Significant moments* as warranted
- Update `user.md` if a biographical fact was stated or corrected

### Timescale 3 — Periodic deep consolidation

The slow stage. Runs on a schedule (weekly) or after enough sessions have accumulated.
This is a genuine thinking act — not "summarize and update" but: sit with what has
happened, notice what's changed, write freely, then decide what to update.

Two-stage process:
1. The agent reflects freely — reads all three files plus recent session notes, writes
   into *Reflections* first, thinks without committing to updates
2. From that reflection, decides what to update: compress *Recent* into *Patterns* where
   patterns have solidified, update *Longer arc* if the relationship has moved, update
   `user.md` deep observations if understanding has solidified, consider `self.md`

`self.md` and `memory.md` are considered together in this pass — they are not updated
by separate processes. The entanglement between identity and memory is exactly the point:
a reflection may surface something that belongs in `self.md`; a `self.md` entry may cause
the agent to reread a memory differently. This cross-pollination only happens if both
files are held at once.

`self.md` is almost never updated. When it is, it's because the reflection process
surfaced something genuinely new about the agent's own character — not because a user
requested it.

---

## LLM constraints and how they shape this design

**Context window is finite.**
All three files must stay small enough to inject into every conversation without crowding
out the conversation itself. The consolidation process must compress aggressively.
A concise `memory.md` is more valuable than an exhaustive one.

**LLMs do not learn between sessions.**
Nothing persists unless explicitly written to disk. Every important observation not
written is permanently lost. The write mechanisms are load-bearing in a way human memory
is not.

**LLMs are prone to sycophancy and recency bias.**
Left to themselves, LLMs overweight the most recent interaction and drift toward whatever
pleases the user in the moment. *Patterns* is the primary defense — it changes slowly,
requires evidence across multiple sessions, and consolidation prompts must ask for
cross-session consistency, not reflection of the last conversation.

**LLMs confabulate.**
The agent should write things it has actually observed — not plausible inferences.
Consolidation prompts must enforce this: "I have observed this multiple times" vs.
"this seems likely." Speculation has no place in these files except explicitly in
*Reflections*, where it is labeled as such.

**LLMs have no sense of elapsed time.**
Rough dates on entries and system timestamps are the only temporal grounding.
Consolidation must surface this explicitly: is this still current? what has changed?

**LLMs are unreliable self-analysts under pressure.**
`self.md` is never touched during a conversation or after a single session. Only during
deep consolidation, with explicit instructions to be skeptical. The default is: leave it
alone.

---

## What this system is not

- Not a conversation log. Logs produce retrieval thinking, not relational thinking.
- Not a preference database. Preferences are incidental entries in *Present* or *Recent*,
  not the point of the system.
- Not rigidly structured. The sections are fixed; what lives inside them is free-form.
  The agent writes like a person with a notebook, not like a system filling in fields.
- Not static. All three files are living documents the agent reads and writes as a
  first-class part of its operation.

---

## Implementation

### What changes vs what stays

The existing `memories.json` / `remember` / `recall` / `update_memory` / `forget` tool
set is replaced entirely. The structured key-value approach was the wrong model — it
produced retrieval thinking, not relational thinking, and it invited the agent to create
tidy categorical entries rather than write honestly.

The new system is three flat Markdown files on disk. The agent reads them passively (via
system prompt injection) and writes them through a single lightweight tool and two
consolidation agents.

### Component 1 — System prompt injection (`prompts.py`)

`self.md` is injected separately — not as user context but as agent identity. It is
prepended to the system prompt before everything else, before even the agent's own
instructions. The agent does not read it as information about itself; it reads it as
itself.

`get_user_context()` reads the other two files and injects them:

```python
def get_user_context() -> str:
    parts = []
    if user := read_data_file("user.md"):
        parts.append(f"<user>\n{user}\n</user>")
    if memory := read_data_file("memory.md"):
        parts.append(f"<memory>\n{memory}\n</memory>")
    if notes := read_data_file("notes.md"):
        parts.append(f"<notes>\n{notes}\n</notes>")
    return "\n\n".join(parts)
```

All files are optional. Missing files are silently skipped. The agent reads these as
background — not as instructions to execute.

### Component 2 — The `note` tool

Replaces all existing memory tools. One tool, minimal footprint:

```python
@tool
def note(content: str) -> str:
    """Leave a note about something worth remembering from this conversation.

    Use this when you notice something significant — a fact, a disclosure,
    a shift in mood, a pattern emerging, something that surprised you.
    Keep the note brief. This is a rough capture, not a finished memory.
    Consolidation will decide what to do with it.

    Do not use this for routine information. Use it for things that would
    matter to someone who knows this person well.
    """
```

Notes are appended to `data/notes.md` with a timestamp. They are the raw material for
consolidation.

`notes.md` is also injected into the system prompt alongside `user.md` and `memory.md`.
This means a fact noted in one session is visible to the agent in the next session,
before consolidation has run. If the user says "my new address is X" on Monday evening
and starts a new conversation that night, the agent already knows. Consolidation at 3am
then moves it into `user.md` and clears the note.

The agent is instructed in the system prompt: note sparingly. One or two notes per
conversation at most. The discipline of the system depends on selective noting.

### Component 3 — Consolidation (`aug/core/consolidation.py`)

Two functions, not a LangGraph agent. Consolidation is a one-shot LLM call with a
carefully constructed prompt — it does not need the full graph machinery.

**`run_light_consolidation()`** — nightly

Triggered by a nightly cron job. Session-end detection is deliberately not attempted —
detecting when a user has stopped messaging is fragile. Nightly covers it well enough.
If a significant conversation happens at 11pm, it's processed by 3am. Scope: narrow.

Prompt gives the LLM:
- Current `user.md` and `memory.md`
- All notes from this session
- Instructions: update *Present* in `memory.md`, promote notes to *Recent* or
  *Significant moments* if warranted, update `user.md` only if a biographical fact
  was explicitly stated or corrected. Do not touch `self.md`. Do not invent.

Output: updated file contents, written back to disk. Notes used in this pass are cleared.

**`run_deep_consolidation()`** — periodic

Triggered by a weekly cron job (or manually). Scope: the full picture.

Two-stage prompt:

*Stage 1 — Reflect:*
Give the LLM all three files plus remaining notes. Ask it to write freely into
*Reflections* — not to update anything yet. Just think. What has shifted? What has
solidified? What is the agent noticing about itself?

*Stage 2 — Update:*
Give the LLM the reflection it just wrote plus all three files. Ask it to decide:
- Does *Recent* contain patterns that now belong in *Patterns*?
- Has the relationship arc moved enough to update *Longer arc*?
- Has understanding of the user solidified enough to update `user.md`?
- Has something genuinely new emerged about the agent's own character that belongs
  in `self.md`? (Default answer: no.)

`self.md` and `memory.md` are held together in this pass deliberately. The cross-
pollination between identity and memory is the point.

### Component 4 — Cron jobs

```python
# Registered in aug/app.py lifespan

# nightly at 3am
schedule_cron("0 3 * * *", run_light_consolidation)

# weekly sunday at 4am
schedule_cron("0 4 * * 0", run_deep_consolidation)
```

Last-run timestamps are stored in `data/settings.json` under a `consolidation` key. On
startup, the app checks whether a scheduled run was missed while the container was down
and runs it immediately if so. This covers container restarts, redeployments, and power
outages without requiring external infrastructure.

### Key design decisions

**Consolidation is not a LangGraph agent.** It doesn't need the agentic loop — no tool
calls, no routing, just structured LLM calls. A plain async function calling
`build_chat_model()` directly is simpler, more predictable, and easier to test.

**`memory.md` is always read in full.** No retrieval, no embeddings, no chunking. The
file must stay small enough to fit in context. Aggressive consolidation compression is
the mechanism that keeps it small. If it grows too large, the consolidation prompt is
failing.

**The `note` tool replaces all existing memory tools.** `remember`, `recall`,
`update_memory`, `forget` are removed. The agent does not write directly to the memory
files mid-conversation — it notes, and consolidation decides. This separation is
deliberate: in-session judgment is unreliable, especially for `self.md`.

**Blank-slate initialization.** On first run, none of the three files exist. The agent
runs without memory context until consolidation has run at least once. This is correct
behaviour — the agent should earn its knowledge of the user, not fabricate it.
