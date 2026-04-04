# PRD: SSH Tool with Command Approval

## Problem Statement

The agent can already run bash commands inside its own Docker container via the `run_bash` tool. However, there is no way to run commands on the host machine or on any other remote server. Adding raw SSH access without guardrails would be dangerous — the agent could execute destructive commands on production infrastructure without the user being aware or consenting. The user needs a way to give the agent controlled, auditable SSH access to remote machines, where every command either matches a pre-approved pattern or requires explicit real-time user approval before it runs.

## Solution

Introduce two new tools — `run_ssh` and `list_ssh_targets` — that allow the agent to execute commands on named SSH targets configured in `settings.json`. Before any command runs, a lightweight approval layer checks whether it matches a saved approval rule (regex per target). If no rule matches, the graph pauses via LangGraph's `interrupt()` mechanism and sends the user an approval prompt with inline buttons. The user can approve for this run only, approve permanently (saving an exact-match rule), or deny. The approved-commands list is managed exclusively through these approval flows and a `/approvals` Telegram command — the agent cannot grant itself new permissions.

## User Stories

1. As a user, I want the agent to run a command on a named remote server, so that I can automate server management tasks through conversation.
2. As a user, I want to see a list of configured SSH targets, so that I can tell the agent which machine to use.
3. As a user, I want the agent to discover available SSH targets on its own, so that it doesn't ask me for connection details I've already configured.
4. As a user, I want to be asked for approval before any unapproved command runs on a remote server, so that I stay in control of what executes on my infrastructure.
5. As a user, I want to approve a command for this run only, so that I can allow a one-off operation without permanently expanding the agent's permissions.
6. As a user, I want to permanently approve an exact command on a specific target, so that the agent can run routine operations without interrupting me every time.
7. As a user, I want to deny a command the agent proposes, so that the agent is told clearly the command was not run.
8. As a user, I want approved rules to be stored persistently, so that they survive bot restarts.
9. As a user, I want to view all saved approval rules, so that I can audit what the agent is allowed to do.
10. As a user, I want to revoke a saved approval rule, so that I can remove permissions I no longer want the agent to have.
11. As a user, I want approval prompts to arrive as Telegram messages with inline buttons, so that I can approve or deny with a single tap.
12. As a user, I want the agent to wait indefinitely for my approval response, so that I'm not pressured to respond immediately.
13. As a user, I want to start a new conversation while an approval is pending, so that an unanswered approval doesn't block me from using the bot.
14. As a user, I want stdout and stderr from SSH commands returned to the agent, so that it can reason about command output and report results accurately.
15. As a user, I want clear error messages when an SSH command fails or the connection is refused, so that the agent can diagnose and report the problem honestly.
16. As a user, I want SSH targets defined by name (not inline credentials), so that the agent never handles raw hostnames, users, or key paths in its reasoning.
17. As a user, I want to add a new SSH target to settings without redeploying, so that I can onboard new machines at runtime.
18. As a user, I want the approval mechanism to be extensible to future tools, so that other high-risk tools can reuse the same pattern.
19. As a user, I want command approval to work through the REST API interface as well, so that non-Telegram clients can also participate in approval flows.
20. As a user, I want the agent to use existing v9 agents that include SSH tools alongside all previously available tools, so that I don't lose any existing capabilities when switching to the SSH-enabled agent.

## Implementation Decisions

### Modules to build or modify

**New: SSH tool module (`aug/core/tools/run_ssh.py`)**
- Two tools: `run_ssh(target: str, command: str)` and `list_ssh_targets()`
- `run_ssh` is decorated with `@requires_approval` (applied before `@tool`)
- `list_ssh_targets` reads target names from settings and returns them — no approval needed
- Uses `asyncssh` for async SSH execution; fresh connection per command
- Returns stdout + stderr; exit code logged but surfaced in the return string on failure
- Resolves target name → connection details (host, port, user, key_path) from settings

**New: Approval decorator module (`aug/core/tools/approval.py`)**
- `ApprovalRequest` dataclass: `target`, `command`
- `ApprovalDecision` enum: `APPROVED_ONCE`, `APPROVED_ALWAYS`, `DENIED`
- `@requires_approval` decorator: checks saved rules first; if no match, calls `interrupt(ApprovalRequest(...))` and acts on the returned `ApprovalDecision`
- `is_approved(target, command)` — checks `settings.json` for a matching rule
- `save_approval(target, command)` — writes exact-match rule (escaped as regex) to settings
- `list_approvals()` / `revoke_approval(index)` — for `/approvals` command use
- No LangGraph imports outside of this module

**Modified: BaseInterface (`aug/api/interfaces/base.py`)**
- New abstract method: `request_approval(request: ApprovalRequest, context: ContextT) -> None`
  - Fire-and-forget: sends the prompt, returns immediately
  - The run exits after this call; resume happens via a separate entry point
- `_execute_run` detects interrupt events via `isinstance(interrupt_value, ApprovalRequest)` and calls `request_approval`

**Modified: TelegramInterface (`aug/api/interfaces/telegram.py`)**
- Implements `request_approval`: sends a message with three inline buttons — "Run Once", "Allow Always", "Deny"
- `callback_data` encodes `thread_id` and `ApprovalDecision`
- New callback query handler: decodes decision, resumes the paused graph with `Command(resume=decision)`
- New `/approvals` command handler: lists saved rules with revoke buttons

**Modified: REST interface (if it exists)**
- Implements `request_approval`: returns a pending-approval response to the caller
- New endpoint `POST /chat/approve/{thread_id}` accepts `{"decision": "approved_once" | "approved_always" | "denied"}`

**Modified: Agent registry (`aug/core/registry.py`)**
- `_V9_TOOLS = [*_V7_TOOLS, run_ssh, list_ssh_targets]`
- Full `v9_*` agent set mirroring `v7_*` (same models, no reflexes)

### Settings schema additions

```
tools.ssh.targets = [
  { "name": "homeserver", "host": "192.168.1.10", "port": 22, "user": "admin", "key_path": "/app/data/keys/homeserver.pem" }
]

tools.ssh.approvals = [
  { "target": "homeserver", "pattern": "df\\s.*" },
  { "target": "homeserver", "pattern": "systemctl status .*" }
]
```

### Approval flow (end-to-end)

1. Agent calls `run_ssh(target="homeserver", command="df -h")`
2. `@requires_approval` checks `tools.ssh.approvals` — no matching rule
3. Decorator calls `interrupt(ApprovalRequest(target="homeserver", command="df -h"))`
4. LangGraph checkpoints state; `astream` yields interrupt event and exits
5. `_execute_run` detects `ApprovalRequest`, calls `request_approval(request, context)`
6. Telegram sends message with three inline buttons; run exits
7. User taps "Allow Always" → callback query arrives
8. Callback handler saves rule to settings, resumes graph with `Command(resume=ApprovalDecision.APPROVED_ALWAYS)`
9. Decorator receives decision, executes the SSH command, returns output
10. Agent continues normally

### Key architectural constraints

- `@requires_approval` applied **before** `@tool` so LangGraph wraps the decorated function
- The decorator is the only place `interrupt()` is called — tools never import LangGraph
- Agent cannot call any function that modifies `tools.ssh.approvals` — that path does not exist
- Approval rules use Python `re.search` for matching (not `re.fullmatch`) — patterns are substrings by default; authors should anchor with `^`/`$` where needed

## Testing Decisions

**What makes a good test:** Test external behaviour only — what the function returns or what side effects it produces — not internal implementation details. Mock at the correct boundary: for SSH, mock `asyncssh.connect`; for settings, use a temp file or in-memory dict; for LangGraph interrupt, mock `interrupt()` itself.

**Modules to test:**

- `aug/core/tools/approval.py` — unit tests for:
  - `is_approved` correctly matches/rejects against regex rules per target
  - `is_approved` with wildcard target `"*"` matches any target
  - `save_approval` writes the correct rule to settings
  - `list_approvals` / `revoke_approval` read and mutate settings correctly
  - The decorator calls `interrupt()` when no rule matches
  - The decorator does NOT call `interrupt()` when a rule matches
  - The decorator saves a rule when decision is `APPROVED_ALWAYS`
  - The decorator returns a denial string when decision is `DENIED`

- `aug/core/tools/run_ssh.py` — unit tests for:
  - Successful command returns stdout
  - Non-zero exit code surfaces in return string
  - Connection failure returns a clear error string
  - Unknown target returns a clear error string (not a stack trace)
  - `list_ssh_targets` returns names from settings; returns helpful message when none configured

**Prior art:** See existing tool tests in `tests/` for mocking patterns. The `run_bash` tests are the closest analogue for command-execution tools.

## Out of Scope

- Multi-user authorization (separate per-chat SSH permissions) — all users in `allowed_chat_ids` share the same approval rules
- Pattern-based approval suggestion by the agent — "Allow with broader pattern" is not a v1 approval option
- Persistent SSH connection pools — fresh connection per command only
- SSH key rotation or key generation — keys are managed externally, only the path is stored
- Wildcard target approvals (`target: "*"`) in the UI — can be added manually to settings but not created through approval flows
- Timeout-based auto-denial — the graph waits indefinitely for user response
- Audit log of executed SSH commands beyond what already appears in structured logs

## Further Notes

- `asyncssh` must be added as a project dependency
- Key files are referenced by path inside the container — ensure keys are mounted into the container in production (`docker-compose.prod.yml`)
- To configure an SSH target, add it to `settings.json` under `tools.ssh.targets`:
  ```json
  {
    "tools": {
      "ssh": {
        "targets": [
          {
            "name": "homeserver",
            "host": "192.168.1.10",
            "port": 22,
            "user": "admin",
            "key_path": "/app/data/keys/homeserver.pem"
          }
        ]
      }
    }
  }
  ```
  Then switch to a `v9_*` agent (e.g. via `/version` → `v9_claude`).
- The `@requires_approval` decorator is intentionally SSH-specific in v1. When a second tool needs approval semantics, extract the generic pattern at that point with two concrete examples to generalise from.
- LangGraph's `interrupt()` value survives a bot restart because the checkpoint is persisted in Postgres via `AsyncPostgresSaver`. A pending approval prompt sent before a restart remains resumable after restart, as long as the user still has the Telegram message with buttons.
- The `/approvals` command should display rules with 1-based indices and a "Revoke" inline button per rule, consistent with how `/version` and `/skills` present lists in the existing Telegram interface.
