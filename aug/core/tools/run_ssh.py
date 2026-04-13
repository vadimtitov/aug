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
from pathlib import Path

import asyncssh
from langchain_core.tools import tool

from aug.core.tools.approval import requires_approval
from aug.utils.data import DATA_DIR
from aug.utils.file_settings import SshTarget, load_settings
from aug.utils.ssh import find_target, get_targets

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_SSH_DOWNLOADS_DIR = DATA_DIR / "ssh_downloads"


@tool
@requires_approval(describe=lambda target, command: (target, command))
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

    logger.info("run_ssh target=%s cmd=%.120r", target, command)

    connect_kwargs = _build_connect_kwargs(cfg)

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
    names = [t.name for t in targets]
    return "Configured SSH targets:\n" + "\n".join(f"  • {n}" for n in names)


@tool
@requires_approval(describe=lambda target, remote_path: (target, f"download {remote_path}"))
async def download_ssh_file(target: str, remote_path: str) -> str:
    """Download a file from a named SSH target to the agent's local storage.

    Uses SFTP. The file is saved under /app/data/ssh_downloads/ and the local
    path is returned so it can be referenced in subsequent operations.

    Args:
        target:      Name of the SSH target (e.g. "homeserver").
        remote_path: Absolute path of the file on the remote machine.
    """
    cfg = find_target(target)
    if cfg is None:
        return f"SSH target '{target}' not found. Use list_ssh_targets() to see configured targets."

    max_bytes = load_settings().tools.ssh.max_download_bytes

    connect_kwargs = _build_connect_kwargs(cfg)

    logger.info("download_ssh_file target=%s remote=%s", target, remote_path)

    try:
        async with asyncssh.connect(**connect_kwargs) as conn:
            async with conn.start_sftp_client() as sftp:
                try:
                    stat = await sftp.stat(remote_path)
                except asyncssh.SFTPError as exc:
                    return f"Cannot stat '{remote_path}' on '{target}': {exc}"

                file_size = stat.size
                if file_size is not None and file_size > max_bytes:
                    return (
                        f"File '{remote_path}' on '{target}' is {file_size:,} bytes, "
                        f"which exceeds the download limit of {max_bytes:,} bytes. "
                        "Increase tools.ssh.max_download_bytes in settings to allow larger downloads."  # noqa: E501
                    )

                _SSH_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
                safe_name = remote_path.lstrip("/").replace("/", "__")
                local_path = _SSH_DOWNLOADS_DIR / f"{target}__{safe_name}"

                await sftp.get(remote_path, str(local_path))
    except asyncssh.HostKeyNotVerifiable as exc:
        logger.warning("download_ssh_file host_key_mismatch target=%s error=%r", target, exc)
        return (
            f"Connection to '{target}' refused — host key mismatch. "
            "The server's key has changed. Re-provision the target via /ssh."
        )
    except Exception as exc:
        logger.warning("download_ssh_file failed target=%s error=%r", target, exc)
        return f"File download did NOT complete. Connection to '{target}' failed: {exc}"

    actual_size = Path(local_path).stat().st_size
    logger.info("download_ssh_file done local=%s size=%d", local_path, actual_size)
    return f"Downloaded '{remote_path}' from '{target}' to '{local_path}' ({actual_size:,} bytes)."


@tool
@requires_approval(
    describe=lambda target, local_path, remote_path: (
        target,
        f"upload {local_path} → {remote_path}",
    )
)
async def upload_ssh_file(target: str, local_path: str, remote_path: str) -> str:
    """Upload a local file to a named SSH target via SFTP.

    Args:
        target:      Name of the SSH target (e.g. "homeserver").
        local_path:  Absolute path of the file on the agent's local filesystem.
        remote_path: Absolute destination path on the remote machine.
    """
    cfg = find_target(target)
    if cfg is None:
        return f"SSH target '{target}' not found. Use list_ssh_targets() to see configured targets."

    if not Path(local_path).exists():
        return f"Local file '{local_path}' does not exist."

    connect_kwargs = _build_connect_kwargs(cfg)
    local_size = Path(local_path).stat().st_size

    logger.info(
        "upload_ssh_file target=%s local=%s remote=%s size=%d",
        target,
        local_path,
        remote_path,
        local_size,
    )

    try:
        async with asyncssh.connect(**connect_kwargs) as conn:
            async with conn.start_sftp_client() as sftp:
                await sftp.put(local_path, remote_path)
    except asyncssh.HostKeyNotVerifiable as exc:
        logger.warning("upload_ssh_file host_key_mismatch target=%s error=%r", target, exc)
        return (
            f"Connection to '{target}' refused — host key mismatch. "
            "The server's key has changed. Re-provision the target via /ssh."
        )
    except Exception as exc:
        logger.warning("upload_ssh_file failed target=%s error=%r", target, exc)
        return f"File upload did NOT complete. Connection to '{target}' failed: {exc}"

    logger.info("upload_ssh_file done target=%s remote=%s", target, remote_path)
    return f"Uploaded '{local_path}' ({local_size:,} bytes) to '{target}:{remote_path}'."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_connect_kwargs(cfg: SshTarget) -> dict:
    """Build asyncssh connection kwargs from an SshTarget."""
    kwargs: dict = {
        "host": cfg.host,
        "port": cfg.port,
        "username": cfg.user,
        "client_keys": [cfg.key_path],
        "connect_timeout": 30,
    }
    if not cfg.verify_host:
        kwargs["known_hosts"] = None
    elif cfg.known_hosts:
        kwargs["known_hosts"] = cfg.known_hosts
    return kwargs
