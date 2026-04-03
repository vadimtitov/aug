"""SSH execution tool — runs commands on named remote targets.

Tools:
  run_ssh(target, command)  — execute a command on a named SSH target; requires
                              user approval (via @requires_approval) unless a
                              saved rule already permits it.
  list_ssh_targets()        — list names of configured SSH targets.

Settings schema (tools.ssh.targets):
  [
    {
      "name":     "homeserver",
      "host":     "192.168.1.10",
      "port":     22,
      "user":     "admin",
      "key_path": "/app/data/keys/homeserver.pem"   # key-based auth
      // OR
      "password": "secret"                           # password-based auth
    },
    ...
  ]
"""

import logging

import asyncssh
from langchain_core.tools import tool

from aug.core.tools.approval import requires_approval
from aug.utils.user_settings import get_setting

logger = logging.getLogger(__name__)

_SETTING_PATH = ("tools", "ssh", "targets")
_TIMEOUT = 60


@tool
@requires_approval
async def run_ssh(target: str, command: str) -> str:
    """Execute a shell command on a named SSH target.

    SSH targets are configured in settings under tools.ssh.targets.
    Use list_ssh_targets() first if you are unsure which targets are available.

    Args:
        target:  Name of the SSH target (e.g. "homeserver").
        command: Shell command to run on the remote machine.
    """
    cfg = _find_target(target)
    if cfg is None:
        return (
            f"SSH target '{target}' not found. "
            "Use list_ssh_targets() to see configured targets."
        )

    logger.info("run_ssh target=%s cmd=%.120r", target, command)

    connect_kwargs: dict = {
        "host": cfg["host"],
        "port": int(cfg.get("port", 22)),
        "username": cfg["user"],
        "known_hosts": None,
    }
    if "password" in cfg:
        connect_kwargs["password"] = cfg["password"]
    elif "key_path" in cfg:
        connect_kwargs["client_keys"] = [cfg["key_path"]]
    else:
        return f"SSH target '{target}' has no 'password' or 'key_path' configured."

    try:
        async with asyncssh.connect(**connect_kwargs) as conn:
            result = await conn.run(command, timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("run_ssh connection_failed target=%s error=%r", target, exc)
        return f"SSH command did NOT run. Connection to '{target}' failed: {exc}"

    output = (result.stdout + result.stderr).strip()

    if result.exit_status != 0:
        logger.warning(
            "run_ssh exit_code=%d target=%s stderr=%.200r",
            result.exit_status,
            target,
            result.stderr.strip(),
        )
        if output:
            return f"Command exited with code {result.exit_status}:\n{output}"
        return f"Command exited with code {result.exit_status} (no output)."

    return output or "(no output)"


@tool
def list_ssh_targets() -> str:
    """List the names of all configured SSH targets.

    Returns a plain-text list of target names that can be passed to run_ssh().
    Call this first when you are unsure which machines are available.
    """
    targets: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    if not targets:
        return (
            "No SSH targets configured. "
            "Add targets under tools.ssh.targets in settings."
        )
    names = [t.get("name", "<unnamed>") for t in targets]
    return "Configured SSH targets:\n" + "\n".join(f"  • {n}" for n in names)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_target(name: str) -> dict | None:
    """Return the target config dict for *name*, or None if not found."""
    targets: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    for t in targets:
        if t.get("name") == name:
            return t
    return None
