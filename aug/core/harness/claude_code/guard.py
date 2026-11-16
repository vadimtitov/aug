"""The PreToolUse hook on Claude's own file and shell tools.

Claude's Bash, Read, Write, Edit, NotebookEdit, Glob and Grep pass it first: it refuses
the paths no tool may touch (``blocked_paths``) and rewrites every shell command to run
through hushed, as ``run_bash`` does.  Monitor's commands are checked too.  It runs in
AUG's own process (an SDK hook), so it cannot fail open the way a hook process could.
"""

from pathlib import Path
from typing import Any

from claude_agent_sdk import HookContext, HookMatcher

from aug.core.tools.blocked_paths import command_block_reason, path_block_reason
from aug.core.tools.run_bash import hushed_command

# Claude's tools that read or write files, and the input field holding the path.
_PATH_FIELDS = ("file_path", "path", "notebook_path")
GUARDED_TOOLS = ("Bash", "Monitor", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep")


def guard_hooks() -> dict:
    """The PreToolUse hook on Claude's own file and shell tools."""
    return {"PreToolUse": [HookMatcher(matcher="|".join(GUARDED_TOOLS), hooks=[guard_tool_use])]}


async def guard_tool_use(hook_input: Any, tool_use_id: str | None, context: HookContext) -> dict:
    """Refuse blocked paths; run shell commands through hushed."""
    name = hook_input.get("tool_name", "")
    args = hook_input.get("tool_input") or {}
    if name == "Bash":
        command = args.get("command", "")
        if reason := command_block_reason(command):
            return _deny(reason)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {**args, "command": hushed_command(command)},
            }
        }
    if name == "Monitor":
        # Its output streams line by line, which hushed would hold back until the
        # command exits: checked, not rewritten, so it runs without secrets.
        command = args.get("command")
        if isinstance(command, str) and (reason := command_block_reason(command)):
            return _deny(reason)
        return {}
    for field in _PATH_FIELDS:
        value = args.get(field)
        if isinstance(value, str) and value and (reason := path_block_reason(Path(value))):
            return _deny(reason)
    return {}


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
