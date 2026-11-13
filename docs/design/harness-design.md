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
  `ToolStarted`, `ToolProgress`, `ToolEnded`, `StatusUpdate`, `ApprovalRequested`,
  `ApprovalResolved`. Events inside a tool call carry `parent_id`, the id of that
  call's `ToolStarted`; a subagent's tool calls are `ToolStarted` events whose parent
  is the subagent's call. Tool names are canonical (no harness prefix).
- Every harness maps its client's exceptions onto `HarnessError.kind`:
  `context_full`, `rate_limited`, `auth`, `unavailable`, `step_limit`, `internal`.
  What the user is told per kind is `HARNESS_ERROR_MESSAGES` in `prompts.py`.
- `Message` is `TextPart`s and `FilePart`s (files are saved by the interface and passed
  by path, never as bytes) plus the `Sender`. `TurnContext.reply_to` says where replies
  go; `attended=False` marks pushes and jobs, where nobody can answer an approval.
- The base `run()` logs every turn and tool call and appends it to AUG's own
  transcript, `data/transcripts/<conversation_id>.jsonl`, the same way on every
  harness. The transcript is a record; nothing feeds it back to a model.

## Tools: `ToolContext` (`aug/core/harness/tool_context.py`)

Every turn hands its tools a `ToolContext` (LangGraph: in the run config, read with
`current_tool_context()`).

- `ctx.emit(event)` adds a progress or status event to the turn's stream, under the
  tool that emits it.
- `await ctx.request_approval(request)` asks the user and waits inside the tool, so the
  turn never pauses. The interface shows `ApprovalRequested` and answers it with
  `pending_approvals.resolve(id, decision)` (Telegram buttons `ap:{id}:{o|a|d}`, REST
  `POST /chat/{context_id}/approve`). A request answers once: a second tap is "already
  handled". Unattended turns get `NO_USER` at once; a typed message instead of a tap
  is `REPLIED` and joins the turn; no answer in 30 minutes is `TIMED_OUT`; `/stop`
  cancels the wait. Each returns its own "NOT executed" string, and a tool whose
  approval was not granted ends with `ToolEnded(error=True)`.

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
