"""SSH provisioning utilities — key generation, installation, and target management.

Provisioning flow:
1. Connect to remote with a one-time password.
2. Generate an Ed25519 keypair in KEYS_DIR.
3. Install the public key on the remote via ~/.ssh/authorized_keys.
4. Capture the server's host key in a per-target known_hosts file.
5. Save the target to settings (key_path + known_hosts path, no password).

After provisioning the password is gone — only the key files remain.
"""

import os
import shlex

import asyncssh

from aug.utils.data import DATA_DIR
from aug.utils.user_settings import get_setting, set_setting

KEYS_DIR = DATA_DIR / "keys"
_SETTING_PATH = ("tools", "ssh", "targets")


async def provision_target(
    name: str, host: str, port: int, user: str, password: str
) -> tuple[str, str, str]:
    """Connect with password, generate Ed25519 keypair, install it, capture host key.

    Returns (key_path, known_hosts_path, fingerprint).
    Files are written but the target is NOT saved to settings yet — caller
    should present the fingerprint to the user and call save_target() on confirm
    or cleanup_keys() on abort.

    Raises RuntimeError on any connection or installation failure.
    """
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    key_path = KEYS_DIR / f"{name}.pem"
    known_hosts_path = KEYS_DIR / f"{name}.known_hosts"

    private_key = asyncssh.generate_private_key("ssh-ed25519")
    pub_key_line = private_key.export_public_key("openssh").decode().strip()

    async with asyncssh.connect(
        host=host,
        port=port,
        username=user,
        password=password,
        known_hosts=None,  # password auth only — fingerprint captured below
        connect_timeout=30,
    ) as conn:
        server_host_key = conn.get_server_host_key()
        fingerprint = server_host_key.get_fingerprint()

        # Install public key; idempotent via grep guard
        cmd = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"grep -qF {shlex.quote(pub_key_line)} ~/.ssh/authorized_keys 2>/dev/null "
            f"|| echo {shlex.quote(pub_key_line)} >> ~/.ssh/authorized_keys && "
            "chmod 600 ~/.ssh/authorized_keys"
        )
        result = await conn.run(cmd)
        if result.exit_status != 0:
            raise RuntimeError(
                f"Key installation failed (exit {result.exit_status}): {result.stderr.strip()}"
            )

        # Write per-target known_hosts in OpenSSH format.
        # Non-22 ports require "[host]:port" as the hostname field.
        host_key_line = server_host_key.export_public_key("openssh").decode().strip()
        hostname_field = f"[{host}]:{port}" if port != 22 else host
        known_hosts_path.write_text(f"{hostname_field} {host_key_line}\n")

    # Write private key only after successful provisioning.
    # Set umask to 0o177 so the file is created 0o600 from the start —
    # no window where the key is world-readable.
    old_mask = os.umask(0o177)
    try:
        private_key.write_private_key(str(key_path))
    finally:
        os.umask(old_mask)

    return str(key_path), str(known_hosts_path), fingerprint


def cleanup_keys(name: str) -> None:
    """Remove generated key files for a target (called on provisioning abort)."""
    for path in (KEYS_DIR / f"{name}.pem", KEYS_DIR / f"{name}.known_hosts"):
        path.unlink(missing_ok=True)


def save_target(
    name: str,
    host: str,
    port: int,
    user: str,
    key_path: str,
    known_hosts_path: str,
) -> None:
    """Add or silently overwrite an SSH target in settings."""
    targets: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    targets = [t for t in targets if t.get("name") != name]
    targets.append(
        {
            "name": name,
            "host": host,
            "port": port,
            "user": user,
            "key_path": key_path,
            "known_hosts": known_hosts_path,
        }
    )
    set_setting(*_SETTING_PATH, value=targets)


def remove_target(name: str) -> None:
    """Remove an SSH target by name from settings."""
    targets: list[dict] = get_setting(*_SETTING_PATH, default=[]) or []
    set_setting(*_SETTING_PATH, value=[t for t in targets if t.get("name") != name])


def get_targets() -> list[dict]:
    """Return all configured SSH targets."""
    return get_setting(*_SETTING_PATH, default=[]) or []


def find_target(name: str) -> dict | None:
    """Return the target config dict for *name*, or None if not found."""
    return next((t for t in get_targets() if t.get("name") == name), None)
