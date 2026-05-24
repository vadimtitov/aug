"""Tests for the Telegram rolling tool-status rendering contract.

Covers the pure functions that decide what the user sees in the live tool
status: per-tool label/arg formatting (display.format_tool), the rolling-list
renderer with truncation and nesting (_render_tool_lines), and the streaming
tail-trim (_draft_preview).
"""

from aug.api.interfaces.telegram.interface import (
    _MAX_SUBLINES,
    _MAX_TOOL_ENTRIES,
    _SPINNER,
    _TG_MAX_LEN,
    _draft_preview,
    _render_tool_lines,
    _ToolEntry,
)
from aug.core.tools.display import _ARG_TRUNCATE, format_tool

# --- format_tool ---------------------------------------------------------


def test_format_tool_known_tool_returns_label_and_arg_preview():
    label, preview = format_tool("brave_search", {"query": "weather"})
    assert label == "🔍 Search"
    assert preview == "weather"


def test_format_tool_unknown_tool_falls_back_to_name_and_first_arg():
    label, preview = format_tool("mystery_tool", {"x": "value"})
    assert label == "mystery_tool"
    assert preview == "value"


def test_format_tool_fetch_page_shows_netlocs():
    _, preview = format_tool("fetch_page", {"urls": ["https://example.com/a/b?c=d"]})
    assert preview == "example.com"


def test_format_tool_subagent_truncates_long_prompt():
    long_prompt = "x" * 100
    _, preview = format_tool("run_subagent", {"prompt": long_prompt})
    assert preview == "x" * _ARG_TRUNCATE + "…"


def test_format_tool_truncates_long_first_arg():
    _, preview = format_tool("run_bash", {"command": "y" * 100})
    assert preview == "y" * _ARG_TRUNCATE + "…"


# --- _render_tool_lines: icons -------------------------------------------


def test_render_active_entry_shows_spinner_frame():
    entry = _ToolEntry(run_id="1", label="🔍 Search", args_preview="cats")
    out = _render_tool_lines([entry], spin_tick=0)
    assert out == f"{_SPINNER[0]} 🔍 Search(cats)"


def test_render_done_entry_shows_green_circle():
    entry = _ToolEntry(run_id="1", label="🔍 Search", args_preview="cats", done=True)
    out = _render_tool_lines([entry], spin_tick=3)
    assert out == "🟢 🔍 Search(cats)"


def test_render_errored_entry_shows_red_circle():
    entry = _ToolEntry(run_id="1", label="🔍 Search", args_preview="cats", done=True, error=True)
    out = _render_tool_lines([entry], spin_tick=0)
    assert out == "🔴 🔍 Search(cats)"


def test_render_empty_args_shows_empty_parens():
    entry = _ToolEntry(run_id="1", label="📋 List tasks", args_preview="", done=True)
    assert _render_tool_lines([entry], spin_tick=0) == "🟢 📋 List tasks()"


# --- _render_tool_lines: top-level truncation ----------------------------


def test_render_no_truncation_when_within_limit():
    entries = [
        _ToolEntry(run_id=str(i), label="🔍 Search", args_preview=str(i), done=True)
        for i in range(_MAX_TOOL_ENTRIES)
    ]
    out = _render_tool_lines(entries, spin_tick=0)
    assert "more" not in out
    assert len(out.splitlines()) == _MAX_TOOL_ENTRIES


def test_render_truncates_oldest_with_hidden_count():
    extra = 4
    entries = [
        _ToolEntry(run_id=str(i), label="🔍 Search", args_preview=str(i), done=True)
        for i in range(_MAX_TOOL_ENTRIES + extra)
    ]
    lines = _render_tool_lines(entries, spin_tick=0).splitlines()
    assert lines[0] == f"…+{extra} more"
    # Only the most recent _MAX_TOOL_ENTRIES entries are shown, newest preserved.
    assert len(lines) == _MAX_TOOL_ENTRIES + 1
    assert lines[-1].endswith(f"({_MAX_TOOL_ENTRIES + extra - 1})")


# --- _render_tool_lines: nested sub-lines --------------------------------


def test_render_nests_sub_lines_under_parent():
    entry = _ToolEntry(
        run_id="1",
        label="🤖 Agent",
        args_preview="research X",
        is_subagent=True,
        sub_lines=["🔍 Search(a)", "🌐 Fetch(b)"],
    )
    lines = _render_tool_lines([entry], spin_tick=0).splitlines()
    assert lines[0].endswith("🤖 Agent(research X)")
    assert lines[1] == "   ↳ 🔍 Search(a)"
    assert lines[2] == "   ↳ 🌐 Fetch(b)"


def test_render_caps_sub_lines_and_keeps_header_visible():
    extra = 3
    subs = [f"step {i}" for i in range(_MAX_SUBLINES + extra)]
    entry = _ToolEntry(
        run_id="1", label="🤖 Agent", args_preview="x", is_subagent=True, sub_lines=subs
    )
    lines = _render_tool_lines([entry], spin_tick=0).splitlines()
    # Header is always first and never pushed out by busy children.
    assert lines[0].endswith("🤖 Agent(x)")
    assert lines[1] == f"   ↳ …+{extra} more"
    # The most recent sub-lines survive.
    assert lines[-1] == f"   ↳ step {_MAX_SUBLINES + extra - 1}"


# --- _draft_preview ------------------------------------------------------


def test_draft_preview_returns_short_text_unchanged():
    assert _draft_preview("hello") == "hello"


def test_draft_preview_trims_long_text_to_tail():
    text = "a" * (_TG_MAX_LEN + 500)
    out = _draft_preview(text)
    assert len(out) <= _TG_MAX_LEN
    assert out.startswith("…")
    # Keeps the most recent characters (the tail of the stream).
    assert out.endswith("a")
