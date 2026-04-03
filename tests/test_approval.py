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

# ---------------------------------------------------------------------------
# is_approved
# ---------------------------------------------------------------------------


def test_is_approved_no_rules():
    with patch("aug.core.tools.approval.get_setting", return_value=[]):
        assert not is_approved("homeserver", "df -h")


def test_is_approved_exact_match():
    rules = [{"target": "homeserver", "pattern": re.escape("df -h")}]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        assert is_approved("homeserver", "df -h")


def test_is_approved_wrong_target():
    rules = [{"target": "otherserver", "pattern": re.escape("df -h")}]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        assert not is_approved("homeserver", "df -h")


def test_is_approved_regex_pattern():
    rules = [{"target": "homeserver", "pattern": r"df.*"}]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        assert is_approved("homeserver", "df -h")
        assert is_approved("homeserver", "df --human-readable /var")
        assert not is_approved("homeserver", "rm -rf /")


def test_is_approved_wildcard_target():
    rules = [{"target": "*", "pattern": re.escape("uptime")}]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        assert is_approved("homeserver", "uptime")
        assert is_approved("workstation", "uptime")
        assert not is_approved("homeserver", "rm -rf /")


def test_is_approved_uses_re_search_not_fullmatch():
    # re.search matches substrings — pattern "df" matches "df -h"
    rules = [{"target": "homeserver", "pattern": "df"}]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        assert is_approved("homeserver", "df -h")


# ---------------------------------------------------------------------------
# save_approval
# ---------------------------------------------------------------------------


def test_save_approval_adds_escaped_rule():
    existing: list = []
    saved: list = []

    def fake_get(*path, default=None):
        return list(existing)

    def fake_set(*path, value):
        saved.clear()
        saved.extend(value)

    with (
        patch("aug.core.tools.approval.get_setting", side_effect=fake_get),
        patch("aug.core.tools.approval.set_setting", side_effect=fake_set),
    ):
        save_approval("homeserver", "df -h")

    assert len(saved) == 1
    assert saved[0]["target"] == "homeserver"
    # Pattern must be re.escape of the command
    assert saved[0]["pattern"] == re.escape("df -h")


def test_save_approval_appends_to_existing():
    existing = [{"target": "homeserver", "pattern": re.escape("uptime")}]
    saved: list = []

    def fake_get(*path, default=None):
        return list(existing)

    def fake_set(*path, value):
        saved.clear()
        saved.extend(value)

    with (
        patch("aug.core.tools.approval.get_setting", side_effect=fake_get),
        patch("aug.core.tools.approval.set_setting", side_effect=fake_set),
    ):
        save_approval("homeserver", "df -h")

    assert len(saved) == 2


# ---------------------------------------------------------------------------
# list_approvals
# ---------------------------------------------------------------------------


def test_list_approvals_empty():
    with patch("aug.core.tools.approval.get_setting", return_value=[]):
        assert list_approvals() == []


def test_list_approvals_returns_rules():
    rules = [
        {"target": "homeserver", "pattern": re.escape("df -h")},
        {"target": "workstation", "pattern": r"uptime.*"},
    ]
    with patch("aug.core.tools.approval.get_setting", return_value=rules):
        result = list_approvals()
    assert len(result) == 2
    assert result[0]["target"] == "homeserver"
    assert result[1]["target"] == "workstation"


# ---------------------------------------------------------------------------
# revoke_approval
# ---------------------------------------------------------------------------


def test_revoke_approval_removes_by_index():
    existing = [
        {"target": "homeserver", "pattern": re.escape("df -h")},
        {"target": "workstation", "pattern": r"uptime.*"},
    ]
    saved: list = []

    def fake_get(*path, default=None):
        return list(existing)

    def fake_set(*path, value):
        saved.clear()
        saved.extend(value)

    with (
        patch("aug.core.tools.approval.get_setting", side_effect=fake_get),
        patch("aug.core.tools.approval.set_setting", side_effect=fake_set),
    ):
        revoke_approval(0)

    assert len(saved) == 1
    assert saved[0]["target"] == "workstation"


def test_revoke_approval_out_of_range_raises():
    existing = [{"target": "homeserver", "pattern": re.escape("df -h")}]
    with (
        patch("aug.core.tools.approval.get_setting", return_value=existing),
        patch("aug.core.tools.approval.set_setting"),
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

    async def my_tool(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(my_tool)

    with (
        patch("aug.core.tools.approval.get_setting", return_value=[]),
        patch("aug.core.tools.approval.interrupt", side_effect=fake_interrupt),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert len(interrupt_values) == 1
    assert isinstance(interrupt_values[0], ApprovalRequest)
    assert interrupt_values[0].target == "homeserver"
    assert interrupt_values[0].command == "df -h"
    assert result == "executed"


@pytest.mark.asyncio
async def test_decorator_skips_interrupt_when_approved():
    interrupt_called = []

    async def my_tool(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(my_tool)
    rules = [{"target": "homeserver", "pattern": re.escape("df -h")}]

    with (
        patch("aug.core.tools.approval.get_setting", return_value=rules),
        patch("aug.core.tools.approval.interrupt", side_effect=lambda v: interrupt_called.append(v)),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert not interrupt_called
    assert result == "executed"


@pytest.mark.asyncio
async def test_decorator_saves_rule_on_approved_always():
    saved: list = []

    async def my_tool(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(my_tool)

    with (
        patch("aug.core.tools.approval.get_setting", return_value=[]),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.APPROVED_ALWAYS),
        patch("aug.core.tools.approval.set_setting", side_effect=lambda *a, value: saved.extend(value)),
    ):
        result = await decorated(target="homeserver", command="df -h")

    assert result == "executed"
    assert any(r["target"] == "homeserver" for r in saved)


@pytest.mark.asyncio
async def test_decorator_returns_denial_string_on_denied():
    async def my_tool(target: str, command: str) -> str:
        return "executed"

    decorated = requires_approval(my_tool)

    with (
        patch("aug.core.tools.approval.get_setting", return_value=[]),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.DENIED),
    ):
        result = await decorated(target="homeserver", command="rm -rf /tmp/test")

    assert "denied" in result.lower()
    assert "rm -rf /tmp/test" in result


@pytest.mark.asyncio
async def test_decorator_executes_tool_on_approved_once():
    executed = []

    async def my_tool(target: str, command: str) -> str:
        executed.append((target, command))
        return "output"

    decorated = requires_approval(my_tool)

    with (
        patch("aug.core.tools.approval.get_setting", return_value=[]),
        patch("aug.core.tools.approval.interrupt", return_value=ApprovalDecision.APPROVED_ONCE),
        patch("aug.core.tools.approval.set_setting"),
    ):
        result = await decorated(target="homeserver", command="uptime")

    assert executed == [("homeserver", "uptime")]
    assert result == "output"
