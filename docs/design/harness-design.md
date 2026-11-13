# Harness design

AUG's interfaces (Telegram, REST, pushes, jobs) talk to one contract, `Harness`, and
never to the thing that runs a turn. Today the only harness is `LangGraphHarness`; a
Claude Code harness is planned on the same contract.

## The contract (`aug/core/harness/`)

```python
class Harness(ABC):
    async def run(self, message: Message, context_id: str, turn: TurnContext) -> AsyncIterator[HarnessEvent]
    async def inject(self, message: Message, context_id: str) -> bool   # False when idle
    async def cancel(self, context_id: str) -> bool                     # ends with TurnEnded(cancelled=True)
    async def compact(self, context_id: str) -> bool                    # caller holds the run lock
    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG) -> str
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

## Names

`context_id` is a chat's current session (Telegram's `/clear` starts a new one);
`conversation_id` is the stable chat it belongs to. LangGraph's checkpoint key and
the Postgres columns keep the old name `thread_id`.
