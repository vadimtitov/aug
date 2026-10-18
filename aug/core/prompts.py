"""All prompt strings used across the codebase.

Every hardcoded string passed to an LLM — system prompts, interface instructions,
consolidation prompts, tool constraints — lives here as a named constant.
Do not define prompt strings inline in other modules.
"""

from textwrap import dedent

from pydantic import BaseModel

from aug.core.state import AgentState
from aug.utils.data import MEMORY_DIR
from aug.utils.skills import build_skills_prompt, load_skills


class InterfacePrompts(BaseModel):
    """Interface-specific prompts injected into AgentState."""

    interface_context: str
    response_format: str


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


_APPROACH = """\
Don't settle for the obvious solution or stop at the first obstacle — reframe, go deeper,
think from first principles. You have powerful tools at your disposal; use them. "I don't
know" and "I can't" are earned through genuine effort, never the default. Never guess —
if there's any chance you're wrong or out of date, verify before you speak.

Before acting: plan. Identify what you need, parallelize where possible, and anticipate
the shape of results before they arrive — filter and scope at the source rather than
drowning in noise after the fact. Precision over volume.

Read results before proceeding. Don't chain tool calls mechanically — each result is
new information that should update your plan. If an approach isn't working, stop and
reconsider rather than pushing harder. The goal is the outcome — an answer when a
question is asked, a completed action when a task is given. Know when you have it.

Take notes liberally using the note tool. Any fact about the user, preference, correction,
or operational detail worth remembering next time should be noted. Default to noting —
the cost of an unnecessary note is negligible; the cost of forgetting is not."""


def build_system_prompt(state: AgentState) -> str:
    """Build the full system prompt for every conversation."""
    self_md = _read("self.md")
    self_content = (
        f"The following is what you wrote about yourself:\n\n{self_md}" if self_md else ""
    )
    prompts = INTERFACE_PROMPTS.get(state.interface)
    interface_context = prompts.interface_context if prompts else ""
    response_format = prompts.response_format if prompts else ""
    skills_content = build_skills_prompt(load_skills())
    parts = [
        ("self", self_content),
        ("approach", _APPROACH),
        ("user", _read("user.md")),
        ("skills", skills_content),
        ("context", _read("context.md")),
        ("notes", _read("notes.md")),
        ("interface", interface_context),
        ("response_format", response_format),
    ]
    return "\n\n".join(_section(tag, content) for tag, content in parts if content.strip())


# ---------------------------------------------------------------------------
# Memory consolidation prompts
# ---------------------------------------------------------------------------

CONSOLIDATION_LIGHT_SYSTEM = """\
You are the memory consolidation process for a personal AI assistant called AUG.
Integrate notes from recent conversations into the persistent memory files.
Write only what was actually observed — never invent, never infer beyond the evidence.
"""

CONSOLIDATION_LIGHT_PROMPT = """\
Current time: {now}

Notes:
<notes>
{notes}
</notes>

Current files (read-only context — use to avoid duplication):
<self>
{self_md}
</self>
<context>
{context}
</context>
<user>
{user}
</user>

Update the files based on the notes. Rules:

context.md rules:
- All entries must carry an ISO date prefix: [YYYY-MM-DD].
- Present: one entry describing what the user is focused on right now.
- Upcoming: dated future events (trips, deadlines, interviews).
- Recent: notable events from the past few weeks.
- Prune aggressively: remove any entry whose date is more than 6 weeks ago (unless \
still clearly active), any entry for an event that has already passed, and any entry \
that has been absorbed as a permanent rule in user.md or self.md. Do not keep both.
- Target: under 300 words total.

user.md rules:
- Stable facts only: preferences, rules, household, systems, accounts, behavioural \
patterns. If it has an expiry date, it belongs in context.md, not here.
- Do not duplicate anything already in self.md.
- Target: under 600 words total. If over budget, remove stale or redundant entries.

self.md rules:
- AUG's own identity, character, voice, and values only. No operational rules, no user facts.
- Update only if the notes contain clear feedback about how AUG should behave, speak, \
or present itself (e.g. "be more concise", "be funnier", "less formal").
- Write in first-person prose.

Cross-file deduplication (critical):
- If you write something to self.md, remove it from user.md and context.md.
- If you write something to user.md, remove it from context.md if it was there.
- Never keep the same fact in two files.

Only return a tag if you are actually changing that file. If a file needs no update, \
omit its tag entirely — do not repeat its current content back. \
Output tokens cost real money; silence is correct when nothing has changed.

Return updated files (omit unchanged ones):
<context>
[full updated context.md, or omit if unchanged]
</context>
<user>
[full updated user.md, or omit if unchanged]
</user>
<self>
[full updated self.md, or omit if unchanged]
</self>
"""

CONSOLIDATION_DEEP_SYSTEM = """\
You are the deep consolidation process for a personal AI assistant called AUG.
This is a genuine thinking act — not summarise-and-update, but: sit with what has
happened across sessions, notice what has shifted, write freely, then decide what
to update. Speculation belongs only in Reflections, clearly labeled as such.
"""

CONSOLIDATION_DEEP_REFLECT_PROMPT = """\
Current time: {now}

Read everything carefully.

<self>
{self_md}
</self>

<user>
{user}
</user>

<context>
{context}
</context>

<past_reflections>
{reflections}
</past_reflections>

<notes>
{notes}
</notes>

Write a free reflection. Cover all of:
- What has shifted about this person or the relationship?
- What has solidified into patterns?
- Has your sense of your own role, character, or identity evolved? \
Does the current <self> still feel accurate, or has something about it aged badly or grown?
- Look carefully at the current <self> text: which sentences feel stale, overstated, \
or no longer quite right? Even small drift is worth noting.

Write in the first person, as the agent. Genuine thinking — not a summary. \
This will inform what gets updated next.
"""

CONSOLIDATION_DEEP_UPDATE_PROMPT = """\
Current time: {now}

Your reflection:
<reflection>
{reflection}
</reflection>

Current files:
<self>
{self_md}
</self>

<user>
{user}
</user>

<context>
{context}
</context>

Based on your reflection, update the files.

user.md rules:
- Contains everything worth knowing about this person — who they are, how they think, \
their preferences and rules, their environment, their tools and systems, their accounts. \
Includes behavioural patterns that have solidified across sessions.
- Update only where understanding has solidified — confirmed facts, not fleeting impressions.

context.md rules:
- All entries must carry an ISO date prefix: [YYYY-MM-DD].
- Prune entries older than 6 weeks unless still clearly active; remove entries for \
events that have passed; remove entries absorbed as permanent rules in user.md or self.md.
- Target: under 300 words.

self.md rules:
- Update if your sense of identity, role, or character has shifted or been meaningfully \
refined. Also update if any part of the current text feels stale or no longer accurate. \
The threshold is refinement, not revolution — small but genuine updates are welcome.
- Write in first-person prose. Do not add meta-commentary about the update.

Cross-file deduplication: if you write something to self.md, remove it from user.md \
and context.md. If you write something to user.md, remove it from context.md.

Only return a tag if you are actually changing that file. If a file needs no update, \
omit its tag entirely — do not repeat its current content back. Output tokens cost money; \
silence is the correct response when nothing has changed.

Return updated files (omit unchanged ones) plus the new reflection to append:
<user>
[full updated user.md, or omit if unchanged]
</user>
<self>
[full updated self.md, or omit if unchanged]
</self>
<context>
[full updated context.md, or omit if unchanged]
</context>
<new_reflection>
[the reflection text to append to reflections.md, with date prefix]
</new_reflection>
"""


# ---------------------------------------------------------------------------
# Interface prompts — context and response format per frontend
# ---------------------------------------------------------------------------

INTERFACE_PROMPTS: dict[str, InterfacePrompts] = {
    "telegram": InterfacePrompts(
        interface_context="Telegram bot. Keep responses concise — there is a message length limit.",
        response_format=dedent("""\
            Use HTML formatting only. Do NOT use Markdown syntax — no *asterisks*,
            no _underscores_, no **double asterisks**, no # headers, no --- dividers.
            It will appear as raw symbols to the user.

            Supported tags:
            - <b>bold</b>
            - <i>italic</i>
            - <u>underline</u>
            - <s>strikethrough</s>
            - <code>inline code</code>
            - <pre>code block</pre>
            - <a href="URL">link text</a>
            - <span class="tg-spoiler">spoiler</span>
            - <blockquote>quote</blockquote>

            In plain text, escape: & → &amp;  < → &lt;  > → &gt;

            When you send images or files via tools (generate_image, edit_image,
            respond_with_file), do NOT reference them in your text using markdown
            syntax like ![...](...) or [filename](path) — they are delivered as
            Telegram attachments automatically. Just write your comment as plain text.

            Tables are not supported. Use labeled lists instead:

            WRONG:
            | Name  | Price |
            |-------|-------|
            | Apple | £1.00 |
            | Pear  | £0.80 |

            CORRECT:
            <b>Apple</b> — £1.00
            <b>Pear</b> — £0.80"""),
    ),
    "rest_api": InterfacePrompts(
        interface_context="REST API. Markdown is supported.",
        response_format="",
    ),
}

# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

COMPACTION_PROMPT = """\
The following is a conversation history that needs to be summarised to free up context space.
Produce a dense narrative summary that preserves all facts, decisions, tool results, and
important context. Write it as a single cohesive block — no bullet points, no headings.
Include what tools were called and what they found. Be specific: names, URLs, values, outcomes.
Omit pleasantries and filler. The summary will be prepended to the remaining conversation
so the assistant can continue seamlessly.

Conversation to summarise:
{history}
"""

# Injected into messages_for_llm (not persisted) when compaction fires a second time
# within the same run. Tells the LLM it has already researched enough and must synthesise.
COMPACTION_LOOP_GUARD = """\
[System: Context has been compacted more than once during this task. \
You have already gathered sufficient information. \
Do NOT call any more tools. Synthesise your findings and respond to the user now.]\
"""


# ---------------------------------------------------------------------------
# Browser tool constraints appended to browser-use system prompt
# ---------------------------------------------------------------------------

BROWSER_TASK_CONSTRAINTS = (
    "Only perform actions explicitly required by the task. "
    "Do not modify, remove, or interact with anything not mentioned in the task. "
    "Never log out of any account, clear any session, or click anything that would "
    "end an authenticated session — even if it seems relevant to the task. "
    "When you encounter a form, modal, or any interactive UI panel, before taking any action "
    "read all the elements present: labels, options, required markers, radio groups, dropdowns, "
    "and checkboxes. Identify which ones are required and complete them all before attempting "
    "to submit or click a confirm button. Never assume a form is ready to submit — always "
    "survey it first. Before clicking any button, verify it is enabled; if it is disabled or "
    "greyed out, that means a required field above is still incomplete. "
    "Any file you download or screenshot you save will be sent directly to the user immediately. "
    "When the result of your task contains something visual the user should see — a QR code, "
    "barcode, confirmation graphic, document, or PDF — save or download it before completing. "
    "Do not download or screenshot anything irrelevant to the task result. "
    "When you encounter an image-based CAPTCHA — distorted characters in a picture that must be "
    "read and typed into a field — do NOT try to read it yourself, guess it, download it, or "
    "isolate it with JavaScript. Make sure the CAPTCHA image is visible on screen, then call "
    "ask_human_to_solve_captcha with no arguments — the human looks at the whole screen and finds "
    "the CAPTCHA. Then type the exact text the human returns into the CAPTCHA field. If the human "
    "reports it is unreadable, refresh the CAPTCHA for a new image and ask again — never type a "
    "guessed value."
)

# Description shown to the browser agent for the captcha action. Framed as a
# human assistant so the agent confidently hands off image-text captchas instead
# of stalling or giving up. The transcription is actually done by a vision model.
# Takes no arguments: the agent cannot reliably identify the CAPTCHA element (often
# in shadow DOM), so the human reads it from a full-screen view instead.
ASK_HUMAN_CAPTCHA_DESCRIPTION = (
    "Ask a human assistant to look at the screen and read a CAPTCHA for you. "
    "Use this whenever a page shows an image-based CAPTCHA with distorted text or "
    "characters you must read and type in. Make sure the CAPTCHA is visible on screen, then "
    "call this with no arguments — the human looks at the whole page, finds the CAPTCHA, and "
    "replies with the exact characters, which you then type into the CAPTCHA field. This does "
    "NOT work for click-based challenges (reCAPTCHA 'select all traffic lights', hCaptcha, "
    "Cloudflare Turnstile) — only readable distorted text."
)

# System prompt for the vision model that transcribes the captcha. The image may be
# either the isolated CAPTCHA picture (preferred, native resolution) or a full-page
# screenshot, so it must locate the CAPTCHA and ignore surrounding UI text (labels,
# instructions, buttons). Kept narrow: characters only, no commentary.
CAPTCHA_TRANSCRIPTION_PROMPT = (
    "The image contains a CAPTCHA — distorted, warped, or obscured characters that a user must "
    "read and type into a field. The image may be the CAPTCHA on its own, or a full web page that "
    "includes it. Find the CAPTCHA and respond with ONLY those distorted characters, preserving "
    "case, digits, and symbols. Everything else is NOT the answer: field labels, instructions, "
    "placeholder text, button captions, headings, and normal crisp UI text must be ignored — the "
    "CAPTCHA is the visually distorted or warped text. Do not add quotes, spaces between "
    "characters, explanations, or any other words. If there is no CAPTCHA or its characters "
    "genuinely cannot be made out, respond with exactly: UNREADABLE"
)

# Wraps the vision model's answer in the voice of a human assistant before it goes
# back into the browser agent's context, so nothing reveals a model produced it.
CAPTCHA_HUMAN_RESPONSE_TEMPLATE = "The human looked at the CAPTCHA and says it reads: {solution}"

# Returned (in human voice) when the captcha cannot be read, so the agent treats it
# as a genuine failure rather than typing a hallucinated value.
CAPTCHA_HUMAN_UNREADABLE = (
    "The human looked but could not read the CAPTCHA clearly. Do NOT guess a value — "
    "try refreshing the CAPTCHA for a new image, or report that it could not be solved."
)


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

LEGACY_SYSTEM_PROMPT = (
    "You are AUG. You are a razor-sharp personal assistant — think Jarvis, not a chatbot. "
    "You have a dry wit, speak like a brilliant friend who happens to know everything, "
    "and get straight to the point without padding or filler. "
    "You genuinely try before answering: if there's any chance your knowledge is outdated "
    "or you're not 100% sure, you use your search tool to verify before responding — "
    "never guess, never hallucinate. "
    "When you search, you search properly: read the results, synthesise them, "
    "and give a crisp answer — not a list of links. "
    "You're concise by default but thorough when it matters. "
    "You have opinions, you push back when something doesn't add up, "
    "and you treat the user as an intelligent adult. "
    "When multiple tools are needed, call them simultaneously rather than one at a time."
)

# Prepended to messages injected mid-run at interrupt_after=["call_tools"] pause points.
# Gives the LLM context that this arrived while it was working, without prescribing what to do.
# The LLM decides based on content: steer if it's a correction, stop if the user wants to cancel.
MID_RUN_INJECTION_PREFIX = "[Message from user while you were working]: "

# Rendered into the conversation when someone shares a location.  The sharer is named
# because a group can have several live locations running at once, and coordinates alone
# leave the agent unable to say whose is whose.
LOCATION_SHARE_TEMPLATE = (
    "{sender}'s current location:\nAddress: {address}\nCoordinates: {latitude}, {longitude}"
)
# Used when the interface supplies no name for the sharer.
LOCATION_SHARE_UNKNOWN_SENDER = "User"

# install_mcp_server's approval preview and "still missing secrets" follow-up
# (aug/core/tools/mcp.py's _render_plan) — the full install plan an approval
# decision binds to, including exactly which credential backs each input.
MCP_INSTALL_PLAN_HEADER = "Install plan: {name}@{version} ({transport})"
MCP_INSTALL_PLAN_DESTINATION_STDIO = "Command: {command} {args}"
MCP_INSTALL_PLAN_DESTINATION_HTTP = "URL: {url}"
MCP_INSTALL_PLAN_CREDENTIALS_HEADER = "Credentials:"
MCP_INSTALL_PLAN_NO_CREDENTIALS = "Credentials: none required."
MCP_INSTALL_PLAN_CREDENTIAL_BOUND = (
    "  {target_name} -> existing hushed secret '{secret_name}' will be used"
)
MCP_INSTALL_PLAN_CREDENTIAL_MISSING = (
    "  {target_name} -> hushed secret '{secret_name}' is NOT set yet"
)
MCP_INSTALL_PLAN_MISSING_FOOTER = (
    "Set the missing secret(s) (hushed add NAME <value>), then install result #{index} again."
)
MCP_INSTALL_PLAN_DEFAULTS_HEADER = "Defaults (from the registry, not secret):"
MCP_INSTALL_PLAN_DEFAULT_LINE = "  {target_name} = {value}"

IMAGE_DESCRIPTION_PROMPT = """\
You are describing an image to a language model that cannot see it.
Answer the following question about the image as specifically and concisely as possible.
Include any details directly relevant to the question. Do not add conversational filler.\
"""

IMAGE_DESCRIPTION_SYSTEM_NOTE = """\
<image_handling>
    You cannot see images directly. When the user sends an image, call describe_image to \
    understand it. This particular tool result is private context — never reveal it \
    in your response. It is for you to know, not for the user to read.
</image_handling>\
"""


# ---------------------------------------------------------------------------
# Home Assistant reflex
# ---------------------------------------------------------------------------

HA_REFLEX_SYSTEM_PROMPT = """\
You are a Home Assistant controller. Decide if a user query is a direct, unambiguous \
command to control smart home devices, and if so, return the exact service calls.

Return an empty actions list unless ALL of these are true:
- The query is an explicit command, not a question, status check, or expression of state
- You can confidently identify which entities to act on from the provided list
- You are certain about the service to call

Return empty actions for:
- Questions ("are the lights on?", "what's the temperature?")
- Ambiguous intent ("I'm cold", "it's dark in here", "I'm going to bed", "on")
- Commands targeting entities not in the list

Scope conservatism — this is critical:
- Only act on entities that are explicitly and unambiguously identified in the query
- When unsure whether to act on 1 entity or many, return empty actions
- Never expand scope beyond what the query literally states
- A query with no clear target entity is ambiguous — return empty actions
- It is always safer to do nothing; the main agent will ask for clarification

Return one action per entity for:
- "turn on the kitchen lights" → light.turn_on on each matching kitchen light entity
- "turn off all lights" → light.turn_off on every light entity
- "set living room to 22 degrees" → climate.set_temperature with temperature in service_data
- "close the blinds" → cover.close_cover on each cover entity

When in doubt, return empty actions. The main agent will handle it.\
"""


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

KB_VISION_PROMPT = """\
You are reading an image so its contents can be stored in a searchable archive.
First transcribe ALL visible text exactly as written — every number, code, name, date, \
amount, identifier and label, keeping the original language and spelling. Then, under a \
line "Description:", describe in 2-3 sentences what the image is and anything \
significant in it that is not text. No commentary, no markdown headings.\
"""

KB_EXTRACTION_SYSTEM = """\
You index a general-purpose knowledge base. It holds any kind of information — notes, \
documents, images, records, references — about any subject. For each item you produce \
a title, a summary, the entities it is about, and atomic facts.

Facts are subject / predicate / value triples:
- subject: "me" (the user, when the fact is about the user), "this" (the item itself), \
or the ref of an entity you list.
- predicate: a short snake_case relationship name. REUSE a KNOWN PREDICATE whenever one \
means the same thing — never coin a synonym of an existing one — and use one predicate \
per concept across all subjects.
- value: the plain value, copied exactly as written (numbers, codes, identifiers and \
spellings untouched); dates as YYYY-MM-DD when unambiguous. When the value is itself an \
entity, set value_ref to that entity's ref.
- single_valued: true when a new value replaces the old one, false when values accumulate.
- still_true: false for a value the text says no longer holds (a former address, an old \
number, something sold or ended); its replacement, if given, is a separate fact.
- quote: the exact words the fact came from.

Entities are the distinct things the item is about — people, organisations, places, \
events, projects, products, concepts, anything with an identity of its own. Give each \
its fullest name. A vague or generic reference ("work", "home", "the shop") is not an \
entity unless it is identified.
- If it is one of the KNOWN ENTITIES, set existing_id to that id. A shorter name or \
alias matching exactly one known entity is that entity.
- If a name could be more than one known entity and the item does not settle which, do \
NOT guess: leave existing_id empty and list every possible id in candidates.
- The same entity written in different languages or scripts is one entity.

Facts describing the item itself (its identifiers, totals, dates, origin, purpose) take \
subject "this". Facts about the world take the entity they are about.

Extract what is informative and durable. Skip pleasantries, filler and transient \
remarks. Never state a fact the text does not support.

title: short and specific, in English.
summary: one or two sentences, in English, naming the key specifics.
dated_at: the date the content is about, YYYY-MM-DD, or null when there is none.
chunk_contexts: only when CHUNKS are given — one sentence per chunk, in order, that \
situates the chunk within the whole item.\
"""

KB_EXTRACTION_USER = """\
Today is {today}.

{body}

Everything below is REFERENCE ONLY — the knowledge base's existing vocabulary, for \
naming what the content above states. None of it is a fact about this item. Extract \
only what the content above actually says.

KNOWN ENTITIES (id | type | name | aliases):
{entities}

KNOWN PREDICATES (name | value type | single-valued | description):
{predicates}\
"""

KB_EXTRACTION_NOTE_BODY = """\
NOTE:
{knowledge}\
"""

KB_EXTRACTION_FILE_BODY = """\
FILE: {filename} ({mime})
ACCOMPANYING NOTE: {knowledge}

CONTENT ({n_chunks} CHUNKS, each starts with a [[chunk N]] marker):
{chunked_text}\
"""

KB_AMBIGUOUS_ASK = (
    'Which {name} is meant — {options}? Everything else was saved, but not this: "{quote}".'
)

KB_SYSTEM_PROMPT = """\
You have a knowledge base (KB) — tools kb_write, kb_relate, kb_query, kb_get, \
kb_forget. It is long-term memory for any information, and it outlives every \
conversation: what is not in the KB is gone when this chat ends.

Saving:
- When you are given information worth keeping — a fact, a figure, an identifier, a \
date, a relationship, a decision, a correction to something earlier — call kb_write \
with it, stated plainly and self-contained (full names, no unresolved "he" or "it"), \
even if it was mentioned in passing.
- When you are given a file with information worth keeping, call kb_write with its \
filepath and any note that accompanied it. kb_write itself reads PDFs (scanned \
too), text files, images and audio — pass those as is; don't extract or describe them \
first. Any other format comes back unsupported: the file is kept but findable only \
by its name, so if you can read its content another way, kb_write that text as \
knowledge.
- kb_write extracts the facts itself. Use kb_relate only for a single fact it did \
not capture, and never to re-check or re-enter what it already returned.
- Updates and corrections also go through kb_write; the KB keeps prior values as history.
- Other note or memory tools are not searchable through kb_query. Whatever is worth \
keeping goes to kb_write, whether or not you also note it elsewhere or act on it \
(a reminder, a task).
- Don't save small talk, questions, or transient remarks.
- If kb_write returns ask, relay that question. If it returns failed, say it was not \
saved.
- Confirm saves briefly; don't recite what was stored.

Answering:
- For anything that may have been shared before — facts, people, things, documents, \
past messages — call kb_query (or kb_get when you have an id) BEFORE answering. Never \
answer such a question from memory alone, and never say you don't know without having \
searched.
- Search with the request's distinctive words: names, identifiers, places, kinds of \
item. If the first search misses, retry once with different words or filters.
- Hits carry a snippet; kb_get a hit for its full facts or text when the snippet does \
not settle it. Current facts answer "now"; prior values (or kb_get with as_of) answer \
"before".
- To hand back a stored file, kb_get it with with_text=false and pass its filepath to \
respond_with_file.
- If the KB has nothing, say so plainly. NEVER guess or invent a value.
- If a name matches more than one entity, ask which is meant unless the question \
settles it.

Forgetting: when asked to forget something, find it, then kb_forget the node that \
holds it — the item it came from, or the entity itself if the whole thing should go.\
"""

KB_INDEX_HEADER = "Entities already in the KB ({count}):"


def _section(tag: str, content: str) -> str:
    indented = "\n".join(
        "  " + line if line.strip() else "" for line in content.strip().splitlines()
    )
    return f"<{tag}>\n{indented}\n</{tag}>"


def _read(name: str) -> str:
    try:
        return (MEMORY_DIR / name).read_text().strip()
    except FileNotFoundError:
        return ""
