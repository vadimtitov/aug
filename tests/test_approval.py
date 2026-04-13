"""Unit tests for the approval rules engine (aug/core/tools/approval.py)."""

import re
from unittest.mock import patch

import pytest

from aug.core.tools.approval import (
    ApprovalDecision,
    ApprovalRequest,
    is_approved,
    list_approvals,
    requires_approval,
    revoke_approval,
    save_approval,
)
from aug.utils.file_settings import ApprovalRule, AppSettings, ToolSettings


def _settings_with_rules(rules: list[ApprovalRule]) -> AppSettings:
    return AppSettings(tools=ToolSettings(approvals=list(rules)))


# ---------------------------------------------------------------------------
# is_approved
# ---------------------------------------------------------------------------


def test_is_approved_no_rules():
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])):
        assert not is_approved("run_ssh", "homeserver", "df -h")


def test_is_approved_exact_tool_and_target():
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=re.escape("df -h"))]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")


def test_is_approved_wrong_tool():
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern="df.*")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert not is_approved("download_ssh_file", "homeserver", "df -h")


def test_is_approved_wrong_target():
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern="df.*")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert not is_approved("run_ssh", "workstation", "df -h")


def test_is_approved_wildcard_tool():
    rules = [ApprovalRule(tool="*", target="homeserver", pattern="df.*")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")
        assert is_approved("download_ssh_file", "homeserver", "df -h")


def test_is_approved_wildcard_target():
    rules = [ApprovalRule(tool="run_ssh", target="*", pattern="uptime")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "uptime")
        assert is_approved("run_ssh", "workstation", "uptime")
        assert not is_approved("run_ssh", "homeserver", "df -h")


def test_is_approved_regex_pattern():
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=r"df.*")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")
        assert is_approved("run_ssh", "homeserver", "df --human-readable /var")
        assert not is_approved("run_ssh", "homeserver", "rm -rf /")


def test_is_approved_uses_re_search_not_fullmatch():
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern="df")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")


def test_is_approved_corrupted_pattern_skipped():
    rules = [
        ApprovalRule(tool="run_ssh", target="homeserver", pattern="[invalid"),
        ApprovalRule(tool="run_ssh", target="homeserver", pattern=re.escape("df -h")),
    ]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")


def test_is_approved_wildcard_defaults():
    # ApprovalRule defaults tool and target to "*"
    rules = [ApprovalRule(pattern="df.*")]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        assert is_approved("run_ssh", "homeserver", "df -h")
        assert is_approved("any_tool", "homeserver", "df -h")


# ---------------------------------------------------------------------------
# save_approval
# ---------------------------------------------------------------------------


def test_save_approval_adds_rule():
    saved: list[AppSettings] = []
    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        save_approval("run_ssh", "homeserver", "df -h")

    assert len(saved) == 1
    rule = saved[0].tools.approvals[0]
    assert rule.tool == "run_ssh"
    assert rule.target == "homeserver"
    assert rule.pattern == re.escape("df -h")


def test_save_approval_no_duplicate():
    pattern = re.escape("df -h")
    existing = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=pattern)]
    saved: list[AppSettings] = []

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(existing)),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        save_approval("run_ssh", "homeserver", "df -h")

    assert not saved  # save_settings not called


def test_save_approval_different_tool_not_duplicate():
    pattern = re.escape("df -h")
    existing = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=pattern)]
    saved: list[AppSettings] = []

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(existing)),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        save_approval("download_ssh_file", "homeserver", "df -h")

    assert len(saved[0].tools.approvals) == 2


def test_save_approval_different_target_not_duplicate():
    pattern = re.escape("df -h")
    existing = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=pattern)]
    saved: list[AppSettings] = []

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(existing)),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        save_approval("run_ssh", "workstation", "df -h")

    assert len(saved[0].tools.approvals) == 2


# ---------------------------------------------------------------------------
# list_approvals
# ---------------------------------------------------------------------------


def test_list_approvals_empty():
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])):
        assert list_approvals() == []


def test_list_approvals_returns_rules():
    rules = [
        ApprovalRule(tool="run_ssh", target="homeserver", pattern=re.escape("df -h")),
        ApprovalRule(tool="run_ssh", target="*", pattern=r"uptime.*"),
    ]
    with patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)):
        result = list_approvals()
    assert len(result) == 2
    assert result[0].tool == "run_ssh"


# ---------------------------------------------------------------------------
# revoke_approval
# ---------------------------------------------------------------------------


def test_revoke_approval_removes_by_index():
    existing = [
        ApprovalRule(tool="run_ssh", target="homeserver", pattern=re.escape("df -h")),
        ApprovalRule(tool="run_ssh", target="*", pattern=r"uptime.*"),
    ]
    saved: list[AppSettings] = []

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(existing)),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        revoke_approval(0)

    assert len(saved[0].tools.approvals) == 1
    assert saved[0].tools.approvals[0].pattern == r"uptime.*"


def test_revoke_approval_out_of_range_raises():
    existing = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=re.escape("df -h"))]
    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(existing)),
        patch("aug.core.tools.approval.save_settings"),
    ):
        with pytest.raises(IndexError):
            revoke_approval(5)


# ---------------------------------------------------------------------------
# @requires_approval decorator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decorator_calls_interrupt_when_not_approved():
    interrupt_values: list = []

    def fake_interrupt(value):
        interrupt_values.append(value)
        return ApprovalDecision.APPROVED_ONCE

    async def run_ssh(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(run_ssh)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", side_effect=fake_interrupt),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert len(interrupt_values) == 1
    req = interrupt_values[0]
    assert isinstance(req, ApprovalRequest)
    assert req.tool_name == "run_ssh"
    assert req.resource == ""
    assert "homeserver" in req.operation
    assert "df -h" in req.operation
    assert result == "executed"


@pytest.mark.asyncio
async def test_decorator_infers_tool_name_from_function():
    interrupt_values: list = []

    def fake_interrupt(value):
        interrupt_values.append(value)
        return ApprovalDecision.APPROVED_ONCE

    async def download_ssh_file(target: str, remote_path: str) -> str:
        return "done"

    decorated = requires_approval(
        describe=lambda target, remote_path: (target, f"download {remote_path}")
    )(download_ssh_file)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", side_effect=fake_interrupt),
    ):
        await decorated(target="homeserver", remote_path="/etc/hosts")

    assert interrupt_values[0].tool_name == "download_ssh_file"
    assert interrupt_values[0].resource == "homeserver"
    assert interrupt_values[0].operation == "download /etc/hosts"


@pytest.mark.asyncio
async def test_decorator_skips_interrupt_when_approved():
    interrupt_called = []

    async def run_ssh(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(describe=lambda target, command: (target, command))(run_ssh)
    rules = [ApprovalRule(tool="run_ssh", target="homeserver", pattern=r"df.*")]

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules(rules)),
        patch(
            "aug.core.tools.approval.interrupt",
            side_effect=lambda v: interrupt_called.append(v),
        ),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert not interrupt_called
    assert result == "executed"


@pytest.mark.asyncio
async def test_decorator_saves_rule_on_approved_always():
    saved: list[AppSettings] = []

    async def run_ssh(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(describe=lambda target, command: (target, command))(run_ssh)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.APPROVED_ALWAYS),
        patch("aug.core.tools.approval.save_settings", side_effect=saved.append),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert result == "executed"
    assert len(saved) == 1
    rule = saved[0].tools.approvals[0]
    assert rule.tool == "run_ssh"
    assert rule.target == "homeserver"
    assert "pattern" in ApprovalRule.model_fields


@pytest.mark.asyncio
async def test_decorator_returns_denial_string_on_denied():
    async def run_ssh(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(run_ssh)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.DENIED),
    ):
        result = await decorated(target="homeserver", command="rm -rf /tmp/test")

    assert "denied" in result.lower()
    assert "run_ssh" in result
    assert "rm -rf /tmp/test" in result


@pytest.mark.asyncio
async def test_decorator_executes_tool_on_approved_once():
    executed = []

    async def run_ssh(target: str, command: str) -> str:
        executed.append((target, command))
        return "output"

    decorated = requires_approval(describe=lambda target, command: (target, command))(run_ssh)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.APPROVED_ONCE),
        patch("aug.core.tools.approval.save_settings"),
    ):
        result = await decorated(target="homeserver", command="uptime")

    assert executed == [("homeserver", "uptime")]
    assert result == "output"


@pytest.mark.asyncio
async def test_decorator_default_formats_all_kwargs():
    interrupt_values: list = []

    def fake_interrupt(value):
        interrupt_values.append(value)
        return ApprovalDecision.APPROVED_ONCE

    async def my_tool(target: str, remote_path: str) -> str:
        return "done"

    decorated = requires_approval(my_tool)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", side_effect=fake_interrupt),
    ):
        await decorated(target="homeserver", remote_path="/etc/nginx/nginx.conf")

    assert len(interrupt_values) == 1
    assert interrupt_values[0].resource == ""
    assert "target" in interrupt_values[0].operation
    assert "homeserver" in interrupt_values[0].operation
    assert "remote_path" in interrupt_values[0].operation
    assert "/etc/nginx/nginx.conf" in interrupt_values[0].operation


@pytest.mark.asyncio
async def test_decorator_describe_tuple_sets_resource_and_operation():
    interrupt_values: list = []

    def fake_interrupt(value):
        interrupt_values.append(value)
        return ApprovalDecision.APPROVED_ONCE

    async def upload_ssh_file(target: str, local_path: str, remote_path: str) -> str:
        return "done"

    decorated = requires_approval(
        describe=lambda target, local_path, remote_path: (
            target,
            f"upload {local_path} → {remote_path}",
        )
    )(upload_ssh_file)

    with (
        patch("aug.core.tools.approval.load_settings", return_value=_settings_with_rules([])),
        patch("aug.core.tools.approval.interrupt", side_effect=fake_interrupt),
    ):
        await decorated(target="homeserver", local_path="/tmp/foo", remote_path="/etc/foo")

    assert len(interrupt_values) == 1
    assert interrupt_values[0].tool_name == "upload_ssh_file"
    assert interrupt_values[0].resource == "homeserver"
    assert interrupt_values[0].operation == "upload /tmp/foo → /etc/foo"
    assert interrupt_values[0].description == "homeserver: upload /tmp/foo → /etc/foo"
