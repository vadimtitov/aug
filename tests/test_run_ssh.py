"""Unit tests for aug/core/tools/run_ssh.py."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aug.utils.file_settings import (
    ApprovalRule,
    AppSettings,
    SshTarget,
    SshToolSettings,
    ToolSettings,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_HOME = SshTarget(
    name="homeserver", host="192.168.1.10", port=22, user="admin", key_path="/keys/home.pem"
)
_WORK = SshTarget(
    name="workstation", host="10.0.0.5", port=22, user="vadim", key_path="/keys/work.pem"
)
_APPROVED_ALL = [ApprovalRule(tool="*", target="*", pattern=".*")]
_APPROVED_DOWNLOAD = [ApprovalRule(tool="download_ssh_file", target="*", pattern=r".*")]
_APPROVED_UPLOAD = [ApprovalRule(tool="upload_ssh_file", target="*", pattern=r".*")]

_P_SSH = "aug.utils.ssh.load_settings"
_P_APPROVAL = "aug.core.tools.approval.load_settings"
_P_RUN_SSH = "aug.core.tools.run_ssh.load_settings"
_P_CONNECT = "aug.core.tools.run_ssh.asyncssh.connect"


def _settings(targets=None, approvals=None, max_download_bytes=None) -> AppSettings:
    ssh = SshToolSettings(
        targets=targets or [],
        max_download_bytes=max_download_bytes or 1_073_741_824,
    )
    return AppSettings(tools=ToolSettings(ssh=ssh, approvals=approvals or []))


# ---------------------------------------------------------------------------
# list_ssh_targets
# ---------------------------------------------------------------------------


def test_list_ssh_targets_no_targets_configured():
    with patch(_P_SSH, return_value=_settings()):
        from aug.core.tools.run_ssh import list_ssh_targets

        result = list_ssh_targets.invoke({})

    assert "no ssh targets" in result.lower() or "not configured" in result.lower()


def test_list_ssh_targets_returns_names():
    with patch(_P_SSH, return_value=_settings(targets=[_HOME, _WORK])):
        from aug.core.tools.run_ssh import list_ssh_targets

        result = list_ssh_targets.invoke({})

    assert "homeserver" in result
    assert "workstation" in result


# ---------------------------------------------------------------------------
# run_ssh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_ssh_unknown_target_returns_error():
    with (
        patch(_P_SSH, return_value=_settings()),
        patch(_P_APPROVAL, return_value=_settings(approvals=[ApprovalRule(pattern=".*")])),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "unknown", "command": "df -h"})

    assert (
        "unknown" in result.lower()
        or "not found" in result.lower()
        or "no ssh target" in result.lower()
    )


@pytest.mark.asyncio
async def test_run_ssh_successful_command():
    mock_result = MagicMock()
    mock_result.stdout = "Filesystem      Size\n/dev/sda1        50G\n"
    mock_result.stderr = ""
    mock_result.exit_status = 0

    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_ALL)),
        patch(_P_CONNECT, return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "df -h"})

    assert "Filesystem" in result
    assert "50G" in result


@pytest.mark.asyncio
async def test_run_ssh_nonzero_exit_code_includes_stderr():
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "bash: badcmd: command not found"
    mock_result.exit_status = 127

    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_ALL)),
        patch(_P_CONNECT, return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "badcmd"})

    assert "command not found" in result or "exit" in result.lower() or "127" in result


@pytest.mark.asyncio
async def test_run_ssh_connection_failure_returns_clear_error():
    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_ALL)),
        patch(
            "aug.core.tools.run_ssh.asyncssh.connect",
            side_effect=OSError("Connection refused"),
        ),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "uptime"})

    assert "failed" in result.lower() or "error" in result.lower() or "connection" in result.lower()
    assert "uptime" not in result


@pytest.mark.asyncio
async def test_run_ssh_empty_output_returns_no_output_marker():
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.exit_status = 0

    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_ALL)),
        patch(_P_CONNECT, return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "true"})

    assert result
    assert "(no output)" in result


# ---------------------------------------------------------------------------
# download_ssh_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_ssh_file_unknown_target_returns_error():
    with (
        patch(_P_SSH, return_value=_settings()),
        patch(_P_APPROVAL, return_value=_settings(approvals=[ApprovalRule(pattern=".*")])),
    ):
        from aug.core.tools.run_ssh import download_ssh_file

        result = await download_ssh_file.ainvoke({"target": "unknown", "remote_path": "/etc/hosts"})

    assert "not found" in result.lower() or "unknown" in result.lower()


@pytest.mark.asyncio
async def test_download_ssh_file_exceeds_size_limit(tmp_path):
    mock_stat = MagicMock()
    mock_stat.size = 2_000_000_000  # 2 GB

    mock_sftp = AsyncMock()
    mock_sftp.stat = AsyncMock(return_value=mock_stat)
    mock_sftp.__aenter__ = AsyncMock(return_value=mock_sftp)
    mock_sftp.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_DOWNLOAD)),
        patch(_P_CONNECT, return_value=mock_conn),
        patch(_P_RUN_SSH, return_value=_settings(max_download_bytes=100_000_000)),
    ):
        from aug.core.tools.run_ssh import download_ssh_file

        result = await download_ssh_file.ainvoke(
            {"target": "homeserver", "remote_path": "/var/log/huge.log"}
        )

    assert "exceeds" in result.lower() or "limit" in result.lower()


@pytest.mark.asyncio
async def test_download_ssh_file_stat_failure_returns_error():
    import asyncssh as _asyncssh

    mock_sftp = AsyncMock()
    mock_sftp.stat = AsyncMock(side_effect=_asyncssh.SFTPError(1, "no such file"))
    mock_sftp.__aenter__ = AsyncMock(return_value=mock_sftp)
    mock_sftp.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_DOWNLOAD)),
        patch(_P_CONNECT, return_value=mock_conn),
        patch(_P_RUN_SSH, return_value=_settings()),
    ):
        from aug.core.tools.run_ssh import download_ssh_file

        result = await download_ssh_file.ainvoke(
            {"target": "homeserver", "remote_path": "/no/such/file"}
        )

    assert "cannot stat" in result.lower() or "no such file" in result.lower()


@pytest.mark.asyncio
async def test_download_ssh_file_success(tmp_path):
    remote_content = b"server config content"
    mock_stat = MagicMock()
    mock_stat.size = len(remote_content)

    async def fake_get(remote, local):
        Path(local).write_bytes(remote_content)

    mock_sftp = AsyncMock()
    mock_sftp.stat = AsyncMock(return_value=mock_stat)
    mock_sftp.get = AsyncMock(side_effect=fake_get)
    mock_sftp.__aenter__ = AsyncMock(return_value=mock_sftp)
    mock_sftp.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_DOWNLOAD)),
        patch(_P_CONNECT, return_value=mock_conn),
        patch(_P_RUN_SSH, return_value=_settings()),
        patch("aug.core.tools.run_ssh._SSH_DOWNLOADS_DIR", tmp_path),
    ):
        from aug.core.tools.run_ssh import download_ssh_file

        result = await download_ssh_file.ainvoke(
            {"target": "homeserver", "remote_path": "/etc/nginx/nginx.conf"}
        )

    assert "homeserver" in result
    assert "/etc/nginx/nginx.conf" in result
    assert str(tmp_path) in result
    assert str(len(remote_content)) in result


@pytest.mark.asyncio
async def test_download_ssh_file_connection_failure_returns_clear_error():
    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_DOWNLOAD)),
        patch(
            "aug.core.tools.run_ssh.asyncssh.connect",
            side_effect=OSError("Connection refused"),
        ),
        patch(_P_RUN_SSH, return_value=_settings()),
    ):
        from aug.core.tools.run_ssh import download_ssh_file

        result = await download_ssh_file.ainvoke(
            {"target": "homeserver", "remote_path": "/etc/hosts"}
        )

    assert "did not complete" in result.lower() or "failed" in result.lower()


# ---------------------------------------------------------------------------
# upload_ssh_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_ssh_file_unknown_target_returns_error():
    with (
        patch(_P_SSH, return_value=_settings()),
        patch(_P_APPROVAL, return_value=_settings(approvals=[ApprovalRule(pattern=".*")])),
    ):
        from aug.core.tools.run_ssh import upload_ssh_file

        result = await upload_ssh_file.ainvoke(
            {"target": "unknown", "local_path": "/tmp/foo", "remote_path": "/etc/foo"}
        )

    assert "not found" in result.lower() or "unknown" in result.lower()


@pytest.mark.asyncio
async def test_upload_ssh_file_missing_local_file_returns_error():
    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_UPLOAD)),
    ):
        from aug.core.tools.run_ssh import upload_ssh_file

        result = await upload_ssh_file.ainvoke(
            {
                "target": "homeserver",
                "local_path": "/nonexistent/path/file.txt",
                "remote_path": "/etc/file.txt",
            }
        )

    assert "does not exist" in result.lower() or "not found" in result.lower()


@pytest.mark.asyncio
async def test_upload_ssh_file_success(tmp_path):
    local_file = tmp_path / "config.txt"
    local_file.write_bytes(b"hello world")

    mock_sftp = AsyncMock()
    mock_sftp.put = AsyncMock(return_value=None)
    mock_sftp.__aenter__ = AsyncMock(return_value=mock_sftp)
    mock_sftp.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_UPLOAD)),
        patch(_P_CONNECT, return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import upload_ssh_file

        result = await upload_ssh_file.ainvoke(
            {
                "target": "homeserver",
                "local_path": str(local_file),
                "remote_path": "/etc/config.txt",
            }
        )

    assert "homeserver" in result
    assert "/etc/config.txt" in result
    assert "11" in result
    mock_sftp.put.assert_awaited_once_with(str(local_file), "/etc/config.txt")


@pytest.mark.asyncio
async def test_upload_ssh_file_connection_failure_returns_clear_error(tmp_path):
    local_file = tmp_path / "file.txt"
    local_file.write_bytes(b"data")

    with (
        patch(_P_SSH, return_value=_settings(targets=[_HOME])),
        patch(_P_APPROVAL, return_value=_settings(approvals=_APPROVED_UPLOAD)),
        patch(
            "aug.core.tools.run_ssh.asyncssh.connect",
            side_effect=OSError("Connection refused"),
        ),
    ):
        from aug.core.tools.run_ssh import upload_ssh_file

        result = await upload_ssh_file.ainvoke(
            {
                "target": "homeserver",
                "local_path": str(local_file),
                "remote_path": "/etc/file.txt",
            }
        )

    assert "did not complete" in result.lower() or "failed" in result.lower()
