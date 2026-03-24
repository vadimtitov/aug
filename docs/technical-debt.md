# Technical Debt

## Performance — `interrupt_after=["call_tools"]` on all agents

**Where:** `aug/core/agents/base_agent.py` — graph compiled with `interrupt_after=["call_tools"]`.

**What it does:** After every tool call, LangGraph pauses the graph and writes a checkpoint to PostgreSQL. This is what enables mid-run message injection (the agent loop in `base.py:_agent_stream` checks the queue at each pause point and injects queued user messages as `HumanMessage`s before resuming).

**The problem:** The pause + checkpoint happens for *every* tool call on *every* agent, including fast tools like `get_current_datetime`, `note`, `brave_search`. For a simple web search + response, this adds a DB write + round-trip per tool call with zero benefit for most interactions.

**Potential fix:** Make `interrupt_after` opt-in per agent, or only enable it for agents that have long-running tools (browser). Alternatively, profile the actual overhead first — it may be negligible at current load.

---

## Correctness — browser tool consumes the injection queue

**Where:** `aug/core/tools/browser.py:_step_callback` — drains `run.pending_agent_injection` and calls `agent.add_new_task()`.

**What happens:** When a user sends a message mid-browser-run, `base.py:run()` puts it in `run.pending_agent_injection`. The browser's step callback drains the queue and injects the message into the browser agent via `add_new_task()`. The browser adapts — but the queue is now empty. When the browser tool finishes and the agent loop checks the queue for leftover messages (in `_agent_stream`), there's nothing there. The main agent never sees those user messages in its conversation history.

**Consequence:** The main agent acted on a user message it has no record of receiving. It can't acknowledge it, can't reference it, and conversation replay/debugging is incomplete.

**Proposed fix (discussed, not implemented):** Add `injections: list[MessageContent]` to `AgentRun`. `inject_message()` appends to both the list AND the queue. The browser reads from the list non-destructively using a local `seen` index (closure variable). The queue stays intact for the agent loop. This is a fan-out / event-log pattern with independent read pointers per consumer.

```python
# AgentRun addition:
injections: list[MessageContent] = field(default_factory=list)

def inject_message(self, content: MessageContent) -> None:
    self.injections.append(content)
    self.pending_agent_injection.put_nowait(content)

# In browser _step_callback (closure):
seen = 0

async def _step_callback(...):
    nonlocal seen
    if run and len(run.injections) > seen:
        new = run.injections[seen:]
        seen = len(run.injections)
        agent_ref[0].add_new_task(_format_injected_messages(new))
```

---

## Correctness — `_stopped_summary` may be obsolete

**Where:** `aug/core/tools/browser.py:_stopped_summary`, called when `run.user_requested_stop.is_set()` after `agent.run()` returns.

**Context:** This was built for a design where new user messages would stop the browser mid-run and return a rich handoff (progress notes, last URL, next goal) so the main agent could resume intelligently. That design was abandoned — new messages are now injected via `add_new_task()` and the browser continues without stopping. The only remaining trigger for `_stopped_summary` is an explicit `/stop` command.

**Question:** Is a rich handoff summary still useful for the explicit-stop case, or is a simple "stopped after N steps" sufficient? If the latter, `_stopped_summary` and all the `model_thoughts()` / `urls()` introspection can be removed.

---

## Scalability — `run_registry` is process-local

**Where:** `aug/core/run.py` — module-level `run_registry = RunRegistry()` singleton.

**The problem:** Each process has its own registry. If the app runs multiple workers (multiple gunicorn workers, multiple Docker replicas behind a load balancer), a `/stop` request or mid-run injection routed to a different process than the one running the agent will silently do nothing. The `POST /{thread_id}/run/cancel` endpoint and Telegram `/stop` command are both broken in this scenario.

**Potential fix:** Replace the in-memory registry with a shared backend. Redis is the natural choice — store run state (active flag, stop signal) as Redis keys with a short TTL. Injection queue becomes a Redis list. This also enables cross-process injection. Non-trivial change; only worth it if horizontal scaling becomes a real requirement.

---

## Docs — README not updated

New features shipped but not documented in README:
- Mid-run message injection (send a message while agent is working → steers it)
- `/stop` command in Telegram + `POST /{thread_id}/run/cancel` REST endpoint
- Browser steering via `add_new_task()` (browser adapts without losing progress)
- Long Telegram responses now split automatically across multiple messages
