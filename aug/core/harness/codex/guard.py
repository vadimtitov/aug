"""The PreToolUse hook on Codex's own shell, patch and image tools.

Codex runs this file before each such call: the call arrives as JSON on stdin, the
decision leaves as JSON on stdout.  It refuses the paths no tool may touch
(``blocked_paths``) and rewrites every shell command to run through hushed, as
``run_bash`` does.  Shell commands from Codex's code mode reach it as ``Bash`` too.
``write_stdin`` never reaches a hook, but it only writes to a process that started
through here.

Codex lets a call through when its hook fails (a crash, output it can't read, its time
limit), so the hook must not fail: a call this file can't judge is refused, it stops
itself before Codex's time limit, and Codex runs it behind ``hook_command()``, which
turns any other failure into a refusal.  It is run as a file, not a module of the
harness package, and imports only what it needs, since every shell command waits for it.
"""

import json
import re
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from aug.core.prompts import CODEX_GUARD_FAILED
from aug.core.tools.blocked_paths import command_block_reason, path_block_reason
from aug.utils.hushed import hushed_command, unhushed

# Codex's tool names as its hooks see them, which AUG's events use too.
SHELL_TOOL = "Bash"
PATCH_TOOL = "apply_patch"
IMAGE_TOOL = "view_image"
MATCHER = f"^({SHELL_TOOL}|{PATCH_TOOL}|{IMAGE_TOOL})$"
# Codex gives a hook this long (its ``timeout``); the guard gives up well before.
HOOK_TIMEOUT_SECONDS = 60
_GIVE_UP_SECONDS = 20
# The file headers of a Codex patch: every path it adds, changes, deletes or moves to.
_PATCH_PATH = re.compile(
    r"^\s*\*\*\*\s*(?:Add File|Update File|Delete File|Move to):\s*(.+?)\s*$", re.M
)


def decide(call: dict[str, Any]) -> dict[str, Any]:
    """The hook's answer to *call*: refused, rewritten, or {} to let it run as is.  A
    call of a shape the guard doesn't know is refused."""
    name = call.get("tool_name")
    args = call.get("tool_input")
    cwd = Path(str(call.get("cwd") or "/"))
    field = {SHELL_TOOL: "command", PATCH_TOOL: "command", IMAGE_TOOL: "path"}.get(str(name))
    value = args.get(field) if isinstance(args, dict) and field else None
    if not isinstance(args, dict) or not isinstance(value, str):
        return _deny(CODEX_GUARD_FAILED)
    if name == SHELL_TOOL:
        if reason := command_block_reason(value):
            return _deny(reason)
        if unhushed(value) is not None:  # already rewritten: a hook that runs twice
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {**args, "command": hushed_command(value)},
            }
        }
    if name == PATCH_TOOL:
        if reason := command_block_reason(value):
            return _deny(reason)
        for path in _PATCH_PATH.findall(value):
            if reason := path_block_reason(cwd / path.strip()):
                return _deny(reason)
        return {}
    if reason := path_block_reason(cwd / value):
        return _deny(reason)
    return {}


def hook_command() -> str:
    """The command Codex runs the guard with: this file, under a shell that turns any
    failure of it into a refusal (exit 2, the reason on stderr)."""
    guard = shlex.join([sys.executable, __file__])
    return f"{guard} || {{ echo {shlex.quote(CODEX_GUARD_FAILED)} >&2; exit 2; }}"


def main() -> None:
    signal.alarm(_GIVE_UP_SECONDS)  # killed by it: a failure, which refuses the call
    try:
        call = json.load(sys.stdin)
        decision = decide(call) if isinstance(call, dict) else _deny(CODEX_GUARD_FAILED)
    except Exception:
        decision = _deny(CODEX_GUARD_FAILED)
    finally:
        signal.alarm(0)
    print(json.dumps(decision))


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


if __name__ == "__main__":
    main()
