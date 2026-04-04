"""SSH execution tool — runs commands on named remote targets.

Tools:
  run_ssh(target, command)  — execute a command on a named SSH target; requires
                              user approval (via @requires_approval) unless a
                              saved rule already permits it.
  list_ssh_targets()        — list names of configured SSH targets.

Targets are provisioned via the /ssh Telegram command, which generates an
Ed25519 keypair and pins the server's host key. Password auth is not supported.

Settings schema (tools.ssh.targets):
  [
    {
      "name":        "homeserver",
      "host":        "192.168.1.10",
      "port":        22,
      "user":        "admin",
      "key_path":    "/app/data/keys/homeserver.pem",
      "known_hosts": "/app/data/keys/homeserver.known_hosts",
      "verify_host": false   # optional escape hatch — disables host verification
    },
    ...
  ]
"""

import logging

import asyncssh
from langchain_core.tools import tool

from aug.core.tools.approval import requires_approval
from aug.utils.ssh import find_target, get_targets

logger = logging.getLogger(__name__)

_TIMEOUT = 60


@tool
@requires_approval
async def run_ssh(target: str, command: str) -> str:
    """Execute a shell command on a named SSH target.

    SSH targets are provisioned via the /ssh Telegram command.
    Use list_ssh_targets() first if you are unsure which targets are available.

    Args:
        target:  Name of the SSH target (e.g. "homeserver").
        command: Shell command to run on the remote machine.
    """
    cfg = find_target(target)
    if cfg is None:
        return f"SSH target '{target}' not found. Use list_ssh_targets() to see configured targets."

    if "key_path" not in cfg:
        return (
            f"SSH target '{target}' has no key_path configured. Re-provision the target via /ssh."
        )

    logger.info("run_ssh target=%s cmd=%.120r", target, command)

    connect_kwargs: dict = {
        "host": cfg["host"],
        "port": int(cfg.get("port", 22)),
        "username": cfg["user"],
        "client_keys": [cfg["key_path"]],
    }

    if cfg.get("verify_host") is False:
        connect_kwargs["known_hosts"] = None
    elif "known_hosts" in cfg:
        connect_kwargs["known_hosts"] = cfg["known_hosts"]
    # else: omit known_hosts → asyncssh uses system default (~/.ssh/known_hosts)

    try:
        async with asyncssh.connect(**connect_kwargs) as conn:
            result = await conn.run(command, timeout=_TIMEOUT)
    except asyncssh.HostKeyNotVerifiable as exc:
        logger.warning("run_ssh host_key_mismatch target=%s error=%r", target, exc)
        return (
            f"Connection to '{target}' refused — host key mismatch. "
            "The server's key has changed. Re-provision the target via /ssh."
        )
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
    targets = get_targets()
    if not targets:
        return "No SSH targets configured. Add targets via the /ssh Telegram command."
    names = [t.get("name", "<unnamed>") for t in targets]
    return "Configured SSH targets:\n" + "\n".join(f"  • {n}" for n in names)
