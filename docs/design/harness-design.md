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
  HarnessError | None`, `cancelled`, `unconsumed` (messages injected too late, which
  the caller runs as a follow-up turn) and, on LangGraph only, `approval` (the turn
  paused on an approval interrupt; the caller asks the user and calls `resume()`).
- Every harness maps its client's exceptions onto `HarnessError.kind`:
  `context_full`, `rate_limited`, `auth`, `unavailable`, `step_limit`, `internal`.
  What the user is told per kind is `HARNESS_ERROR_MESSAGES` in `prompts.py`.
- `Message` is `TextPart`s and `FilePart`s (files are saved by the interface and passed
  by path, never as bytes) plus the `Sender`. `TurnContext.reply_to` says where replies
  go; `attended=False` marks pushes and jobs, where nobody can answer an approval.
- The base `run()` logs every turn and tool call and appends it to AUG's own
  transcript, `data/transcripts/<conversation_id>.jsonl`, the same way on every
  harness. The transcript is a record; nothing feeds it back to a model.

## Who holds which lock (`aug/core/run.py`)

- `run_lock(context_id)`: held for a whole turn by whoever runs one: interfaces
  (including the follow-up turn for unconsumed messages), approval resumes,
  `/compact` and `fire_push`. Two turns never share a context.
- `routing_lock(context_id)`: short. Serialises "inject into the turn in flight, or
  queue a new one". An interface starts its turn under it and holds it until
  `TurnStarted`, so a message arriving in between is injected, not queued behind.

## Names

`context_id` is a chat's current session (Telegram's `/clear` starts a new one);
`conversation_id` is the stable chat it belongs to. LangGraph's checkpoint key and
the Postgres columns keep the old name `thread_id`.

## What is still LangGraph-shaped

Interfaces still render the parsed graph events (`ChatModelStreamEvent`,
`ToolStartEvent`…) and approvals still pause the graph (`TurnEnded.approval` +
`LangGraphHarness.resume()`). Both move behind harness events and
`ToolContext.request_approval()` in the next step, after which no LangGraph type
reaches an interface.
