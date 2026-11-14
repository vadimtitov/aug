# Harness design

AUG's interfaces (Telegram, REST, pushes, jobs) talk to one contract, `Harness`, and
never to the thing that runs a turn. Two harnesses implement it: `LangGraphHarness`
(`v13_kimi`, `v13_glm`) and `ClaudeCodeHarness` (`v14_claude_code`, see below).

## The contract (`aug/core/harness/`)

```python
class Harness(ABC):
    async def run(self, message: Message, context_id: str, turn: TurnContext) -> AsyncIterator[HarnessEvent]
    async def inject(self, message: Message, context_id: str) -> bool   # False when idle
    async def cancel(self, context_id: str) -> bool                     # ends with TurnEnded(cancelled=True)
    async def compact(self, context_id: str) -> bool                    # caller holds the run lock
    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG, *, system: str = "") -> str
    def running(self, on_unsolicited) -> AsyncContextManager             # entered once by the lifespan
```

- `run()` never raises. Its first event is `TurnStarted` (from then on `inject()`
  reaches the turn), its last is always one `TurnEnded`, carrying `error:
  HarnessError | None`, `cancelled` and `unconsumed` (messages injected too late,
  which the caller runs as a follow-up turn).
- In between come harness events, the only thing interfaces render: `TextDelta`,
  `ToolStarted`, `ToolProgress`, `ToolEnded`, `FileEmitted`, `StatusUpdate`,
  `ApprovalRequested`, `ApprovalResolved`. Events inside a tool call carry `parent_id`, the id of that
  call's `ToolStarted`; a subagent's tool calls are `ToolStarted` events whose parent
  is the subagent's call. Tool names are canonical (no harness prefix).
- Every harness maps its client's exceptions onto `HarnessError.kind`:
  `context_full`, `rate_limited`, `auth`, `unavailable`, `step_limit`, `internal`.
  What the user is told per kind is `HARNESS_ERROR_MESSAGES` in `prompts.py`.
- `Message` is `TextPart`s and `FilePart`s (files are saved by the interface and passed
  by path, never as bytes; see Files below) plus the `Sender`. `TurnContext.reply_to` says where replies
  go; `attended=False` marks pushes and jobs, where nobody can answer an approval.
- `oneshot()` is a bare model call (no tools, no history) for jobs. The caller asks for
  a tier; each harness maps tiers to its own models (`LangGraphHarness(models=...)`, from
  `_TIER_MODELS` in the registry).
- The base `run()` logs every turn and tool call and appends it to AUG's own
  transcript, `data/transcripts/<conversation_id>.jsonl`, the same way on every
  harness. Files appear in it as path, media type, filename, size and sha256, never
  bytes. The transcript is a record; nothing feeds it back to a model.

## Tools: `ToolContext` (`aug/core/harness/tool_context.py`)

Every turn hands its tools a `ToolContext` (LangGraph: in the run config, read with
`current_tool_context()`).

- `ctx.emit(event)` adds a progress or status event to the turn's stream, under the
  tool that emits it.
- `ctx.send_file(stored, caption)` sends the user a file already in the outbox, as a
  `FileEmitted` under the tool. Tool results carry text only.
- `await ctx.request_approval(request)` asks the user and waits inside the tool, so the
  turn never pauses. The interface shows `ApprovalRequested` and answers it with
  `pending_approvals.resolve(id, decision)` (Telegram buttons `ap:{id}:{o|a|d}`, REST
  `POST /chat/{context_id}/approve`). A request answers once: a second tap is "already
  handled". Unattended turns get `NO_USER` at once; a typed message instead of a tap
  is `REPLIED` and joins the turn; no answer in 30 minutes is `TIMED_OUT`; `/stop`
  cancels the wait. Each returns its own "NOT executed" string, and a tool whose
  approval was not granted ends with `ToolEnded(error=True)`.

## Files (`aug/utils/storage.py`, `aug/utils/media.py`)

Every file lives in the workspace, `data/workspace/files/`, which survives `/clear`:
`inbox/YYYY-MM/` for what the user sent, `outbox/YYYY-MM/` for a snapshot of each file
sent to the user, and `index/<id>.json` with its metadata.

- **In.** The interface saves each file to the inbox (`store_file`) and passes a
  `FilePart(path, media_type, filename)`. Images are normalised on the way in: HEIC and
  anything over 1568 px on the long edge become a JPEG at most 1568 px. A voice note
  gets its Whisper transcript as a `TextPart` next to the audio `FilePart`; a location
  becomes a `TextPart`. Telegram groups an album (one update per item, sharing a
  `media_group_id`) into one message, together with any message sent alongside it,
  and takes video, video notes and animations;
  REST resolves `ChatRequest.attachments` (ids from `POST /files/upload`).
- **LangGraph.** Every `FilePart` is rendered with a hint carrying its path
  (`[Image from user: <path>]`, `[File from user: …]`), so the agent can hand it to
  tools. Images are stored in history as `[[img:path|mime]]` markers and every model
  call expands all of them the same way, re-reading each from its path: the image
  itself plus the hint when the model has vision, the hint alone (and
  `describe_image`) when it hasn't. The prompt prefix stays byte-identical turn to
  turn, so it stays cached; stored files are read-only so an image can't change under
  it. Only images in the file store, of a type models accept, are inlined (anything
  else a marker names gets the hint), at most the newest 50 per request, and
  compaction counts each image at about 1.6k tokens.
- **Out.** A file reaches the user only as a `FileEmitted`. `respond_with_file(path,
  caption)` is the tool for any file on disk: it refuses the paths in
  `aug/core/tools/blocked_paths.py` (shared with `run_bash`), snapshots the file into
  the outbox and emits. `generate_image`, `edit_image` and the skills tools emit the
  same way, and so do a subagent's tools. Telegram sends by media type (photo, voice,
  audio, video, animation, else a document, which is also the fallback when Telegram
  refuses the first form); a live chat sends at once, a push after its text. REST
  streams `event: file` with the file's id and url, and `GET /files/{id}/content`
  serves it.

## Who holds which lock (`aug/core/run.py`)

- `run_lock(context_id)`: held for a whole turn by whoever runs one: interfaces
  (including the follow-up turn for unconsumed messages), `/compact` and
  `fire_push`. Two turns never share a context.
- `routing_lock(context_id)`: short. Serialises "inject into the turn in flight, or
  queue a new one". An interface starts its turn under it and holds it until
  `TurnStarted`, so a message arriving in between is injected, not queued behind.

## Claude Code (`aug/core/harness/claude_code*.py`)

`ClaudeCodeHarness` runs a `ClaudeCodeProfile` (model, effort, Claude's built-in tools,
AUG's tools, oneshot models) on the Claude Agent SDK. `claude-agent-sdk` is pinned, and
the CLI comes inside it, so both move together.

- **Processes.** Each chat gets its own CLI process (`ChatSession`), opened on first use
  and kept warm. The evictor closes idle ones after an hour (the prompt cache's
  lifetime) and keeps at most 8 idle; a process with a turn in flight or background
  work running is never closed. A closed session resumes from Claude's own session file
  on the next message.
- **Records.** `/app/data/claude/aug-sessions/` keeps one record per `context_id`: the
  Claude session id, the address, and the prompt and skills the session started with.
  A resumed session gets them back unchanged, so `/prompt` and the cache hold across
  restarts. `CLAUDE_CONFIG_DIR` is `/app/data/claude`, so Claude's session files survive
  a redeploy.
- **Setup changes.** Each process is fingerprinted (prompt, skills, model, effort,
  tools, CLI settings). When the fingerprint changes (an MCP server added, say), the
  next turn restarts the process and resumes.
- **Prompt.** Claude Code's own prompt plus AUG's session prompt (`append`), without
  the skills index. `setting_sources=[]`: nothing loads from `~/.claude` or the
  workspace's `.claude/`. The workspace loads as a local plugin named `aug`, and the
  session's skills are passed explicitly (`aug:<name>`), which also hides the CLI's own
  bundled skills.
- **Tools.** AUG's tools (the MCP manager's included) reach Claude through one
  in-process MCP server, `aug`; each call runs the LangChain tool with the turn's
  `ToolContext`, so tools don't know which harness runs them. Events use canonical
  names (`mcp__aug__` stripped). Claude's own tools are an allowlist
  (`_CLAUDE_BUILTIN_TOOLS`) and run without approval (`permission_mode="dontAsk"`):
  anything not listed, including a tool a new CLI adds, is refused. Approvals stay on
  AUG's tools that ask for them.
- **Guard.** A PreToolUse hook refuses the blocked paths (`blocked_paths`) in Bash,
  Read, Write and Edit, and rewrites every shell command to run through hushed, as
  `run_bash` does. The CLI starts through `claude_cli.sh`, which keeps only the
  allowlisted environment variables (`_ENV_KEEP`): AUG's own secrets never reach it.
  The login is the `CLAUDE_CODE_OAUTH_TOKEN` secret in hushed (`claude setup-token`).
- **Turns.** One reader per process maps the SDK's messages onto harness events. A
  turn ends at its result followed by the CLI going idle, so a message injected late
  still belongs to it. When background work finishes after the turn, the CLI starts a
  turn by itself; it is delivered through `running(on_unsolicited)` to
  `dispatch.deliver_unsolicited`, which sends it like an agent push under the chat's
  run lock. Claude's cron is off: scheduling stays with `create_task` and
  `set_reminder`.
- **Errors.** Assistant errors, rate-limit events and error results map onto
  `HarnessError` kinds; a rejected rate limit carries its reset time, and a warning
  becomes a status line.
- **`oneshot()`** is a bare query from a neutral folder: no tools, no settings, one turn.

Tests: `tests/test_claude_code.py` scripts the SDK's messages; `tests/test_claude_code_cli.py`
runs the real CLI against a fake Messages API (`tests/claude_fake_api.py`), free.

## Names

`context_id` is a chat's current session (Telegram's `/clear` starts a new one);
`conversation_id` is the stable chat it belongs to. LangGraph's checkpoint key and
the Postgres columns keep the old name `thread_id`.

## Jobs (`aug/core/jobs.py`)

A job is an async function that receives a harness. `JOBS` is the one list of what runs
when (UTC crontab) and on which profile, so each job runs once whichever agent versions
are live:

```python
JOBS = [
    JobSpec(job="dream_light", cron="0 3 * * *", profile="v13_kimi"),
    JobSpec(job="dream_deep", cron="0 4 * * sun", profile="v13_kimi"),
]
```

`job_runner` schedules them at startup and, in the background, runs any job whose
scheduled time passed since its last run (`state.json` `jobs`). `/consolidate` and
`/consolidate_deep` call `job_runner.run()`; one run of a job at a time. Scheduled tasks
and reminders are not jobs: they belong to a chat and run on its profile through
`fire_push`.
