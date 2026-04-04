# AUG Architecture Friction Report

Generated: 2026-04-03

---

## 1. `BaseInterface` god class
**File:** `aug/api/interfaces/base.py` (458 lines)

Handles message preprocessing, orchestration, streaming, reflex injection, history extraction — all in one class. Subclasses only implement 3–4 methods but inherit the entire responsibility graph. The streaming loop (`_agent_stream`) and reflex injection are untestable without a live LangGraph.

- **Category:** Shallow interface over complex implementation, coupled concerns
- **Test impact:** Integration tests could replace the current untested orchestration paths

---

## 2. `set_reminder` tool mixing I/O with business logic
**File:** `aug/core/tools/set_reminder.py`

The tool directly opens asyncpg connections, contains regex string manipulation of DSNs (duplicated from `db.py`), reads user settings, and constructs SQL — all inside a tool. Violates the core/utils layer split.

- **Category:** Client logic mixed with orchestration; missing tool/service boundary
- **Test impact:** Currently untestable without a real DB; a service boundary enables proper mocking

---

## 3. Gmail auth crosses the layer boundary
**File:** `aug/core/tools/gmail.py`

Core tools import from `aug.api.routers.gmail_auth` (load_token, save_token). Core must not know about API routers.

- **Category:** Inverted dependency (core depends on api)
- **Test impact:** Tests of Gmail tools require mocking router internals

---

## 4. `run_registry` module-level singleton
**Files:** `aug/core/run.py`, `aug/api/interfaces/base.py`, `aug/api/routers/chat.py`

A global singleton tracks in-flight agent runs and is accessed directly from both the router and interface layers with no injection point. Makes multi-tenancy or isolated testing impossible.

- **Category:** Hidden shared state, tight coupling across layers
- **Test impact:** Tests must patch module-level globals

---

## 5. Memory system scattered across 3 files
**Files:** `aug/core/memory.py`, `aug/core/prompts.py`, `aug/app.py`

Understanding memory consolidation requires bouncing between the file I/O code, the prompts that define consolidation rules, and the scheduler startup. Default templates and the prompts that update them can drift out of sync.

- **Category:** Conceptual coupling without co-location
- **Test impact:** Tests must mock across two modules to test consolidation behavior

---

## 6. `browser.py` mutable forward-reference hack
**File:** `aug/core/tools/browser.py`

`agent_ref: list[Agent] = []` is used as a mutable forward reference to break a circular dependency between the tool and the agent it needs to inject into. Fragile and invisible — the injection path is a silent no-op if `agent_ref` isn't populated.

- **Category:** Circular dependency resolved with a code smell
- **Test impact:** The injection path is silently a no-op if agent_ref isn't populated

---

## 7. Reflex injection has non-deterministic semantics
**File:** `aug/api/interfaces/base.py` (lines 186–260)

A reflex finishing mid-run injects into the current run at the next interrupt; a reflex finishing after the run spawns a new run. The timing is non-deterministic and completely untested.

- **Category:** Shared mutable state across async boundaries
- **Test impact:** No tests cover the timing split; bugs here would be subtle and hard to reproduce

---

## 8. `app.state` as implicit, untyped dependency injection
**Files:** `aug/app.py` + `aug/utils/notify.py`, `aug/utils/reminders.py`, routers, interfaces

Seven different subsystems set/read from `app.state` with no formal contract. Initialization order matters but isn't enforced. `notify.py` raises `RuntimeError` hours after startup if an interface failed to register.

- **Category:** Implicit dependencies, latent runtime failures
- **Test impact:** Every test that exercises any router or interface must reconstruct the right `app.state` shape

---

## Systemic patterns

1. **Module-level singletons** (`run_registry`, `_correlation_id`, `_thread_id`) make testing hard and tightly couple layers
2. **Implicit dependency injection via `app.state`** and `ContextVar`s makes initialization order fragile
3. **Shallow interfaces over complex orchestration** (`BaseInterface` 450+ lines) makes subclassing risky
4. **Tools mixing business logic with I/O** (`set_reminder` opening DB connections) makes testing and reuse difficult
5. **Scattered configuration and defaults** (model names hardcoded in 5+ files) makes it hard to change system behavior
6. **Missing abstractions for cross-cutting concerns** (settings, notifications, skills) leads to duplicated integration logic
7. **Untested integration paths** (reflex timing, interrupt loops, OAuth flows) are high-risk
8. **Implicit state shared across async boundaries** (`run.pending_agent_injection` queue, `app.state.interfaces` dict) with no structural enforcement
9. **Data-driven behavior hard-coded** (agent registry, tool display names, interface prompts) requiring code edits for every change
