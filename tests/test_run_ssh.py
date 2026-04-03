"""Unit tests for aug/core/tools/run_ssh.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# list_ssh_targets
# ---------------------------------------------------------------------------

_HOME = {
    "name": "homeserver",
    "host": "192.168.1.10",
    "port": 22,
    "user": "admin",
    "key_path": "/keys/home.pem",
}
_WORK = {
    "name": "workstation",
    "host": "10.0.0.5",
    "port": 22,
    "user": "vadim",
    "key_path": "/keys/work.pem",
}
_APPROVED_ALL = [{"target": "homeserver", "pattern": ".*"}]


def test_list_ssh_targets_no_targets_configured():
    with patch("aug.core.tools.run_ssh.get_setting", return_value=[]):
        from aug.core.tools.run_ssh import list_ssh_targets

        result = list_ssh_targets.invoke({})

    assert "no ssh targets" in result.lower() or "not configured" in result.lower()


def test_list_ssh_targets_returns_names():
    with patch("aug.core.tools.run_ssh.get_setting", return_value=[_HOME, _WORK]):
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
        patch("aug.core.tools.run_ssh.get_setting", return_value=[]),
        patch(
            "aug.core.tools.approval.get_setting",
            return_value=[{"target": "unknown", "pattern": ".*"}],
        ),
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
        patch("aug.core.tools.run_ssh.get_setting", return_value=[_HOME]),
        patch("aug.core.tools.approval.get_setting", return_value=_APPROVED_ALL),
        patch("aug.core.tools.run_ssh.asyncssh.connect", return_value=mock_conn),
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
        patch("aug.core.tools.run_ssh.get_setting", return_value=[_HOME]),
        patch("aug.core.tools.approval.get_setting", return_value=_APPROVED_ALL),
        patch("aug.core.tools.run_ssh.asyncssh.connect", return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "badcmd"})

    assert "command not found" in result or "exit" in result.lower() or "127" in result


@pytest.mark.asyncio
async def test_run_ssh_connection_failure_returns_clear_error():
    with (
        patch("aug.core.tools.run_ssh.get_setting", return_value=[_HOME]),
        patch("aug.core.tools.approval.get_setting", return_value=_APPROVED_ALL),
        patch(
            "aug.core.tools.run_ssh.asyncssh.connect",
            side_effect=OSError("Connection refused"),
        ),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "uptime"})

    assert "failed" in result.lower() or "error" in result.lower() or "connection" in result.lower()
    assert "uptime" not in result  # should NOT claim command ran


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
        patch("aug.core.tools.run_ssh.get_setting", return_value=[_HOME]),
        patch("aug.core.tools.approval.get_setting", return_value=_APPROVED_ALL),
        patch("aug.core.tools.run_ssh.asyncssh.connect", return_value=mock_conn),
    ):
        from aug.core.tools.run_ssh import run_ssh

        result = await run_ssh.ainvoke({"target": "homeserver", "command": "true"})

    assert result  # must not be empty string
    assert "(no output)" in result
