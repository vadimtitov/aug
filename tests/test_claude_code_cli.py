"""ClaudeCodeHarness with the real Claude Code CLI against a fake Messages API.

Free: no Claude account is used.  Covers what a fake SDK client can't: the env wrapper,
the in-process MCP bridge to AUG's tools, the guard hook, and resume from Claude's own
session file.
"""

import json
from typing import Any

import pytest
from langchain_core.tools import tool

from aug.core.harness import (
    Address,
    Message,
    Sender,
    TextDelta,
    TextPart,
    ToolEnded,
    TurnContext,
    TurnEnded,
)
from aug.core.harness import claude_code as cc
from aug.core.harness.claude_code import ClaudeCodeHarness, ClaudeCodeProfile
from aug.core.run import run_registry
from aug.core.tools.context import current_tool_context
from aug.utils.hushed import HushedError
from tests.claude_fake_api import FakeAnthropicAPI, last_user_content, text, tool_use

pytestmark = pytest.mark.skipif(not cc._BUNDLED_CLI.exists(), reason="no bundled Claude CLI")

ADDR = Address(interface="rest_api", conversation_id="chat", chat_id="chat")


@tool
async def echo(value: str) -> str:
    """Echo *value* back, with the chat it ran for."""
    ctx = current_tool_context()
    return f"echo:{value}:{ctx.context_id if ctx else 'none'}"


def _respond(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The scripted model: a marker in the user's text picks what it does; a tool result
    is reported back as text."""
    content = last_user_content(request)
    results = [b for b in content if b.get("type") == "tool_result"]
    if results:
        last_call = _last_tool_call(request)
        if last_call == "ToolSearch":
            return [tool_use("t-echo", "mcp__aug__echo", {"value": "x"})]
        body = results[0]["content"]
        output = body if isinstance(body, str) else " ".join(b.get("text", "") for b in body)
        return [text(f"result: {output}")]
    said = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    names = {t.get("name") for t in request.get("tools", [])}
    if "MARK:echo" in said:
        if "mcp__aug__echo" in names:
            return [tool_use("t-echo", "mcp__aug__echo", {"value": "x"})]
        return [tool_use("t-search", "ToolSearch", {"query": "select:mcp__aug__echo"})]
    if "MARK:bash" in said:
        return [tool_use("t-bash", "Bash", {"command": "echo API_KEY=${API_KEY:-unset}"})]
    if "MARK:secret" in said:
        return [tool_use("t-read", "Read", {"file_path": "/app/data/settings.json"})]
    return [text("hello there")]


def _last_tool_call(request: dict[str, Any]) -> str | None:
    for message in reversed(request.get("messages", [])):
        if message["role"] == "assistant" and isinstance(message["content"], list):
            calls = [b["name"] for b in message["content"] if b.get("type") == "tool_use"]
            if calls:
                return calls[-1]
    return None


@pytest.fixture
def api(monkeypatch):
    with FakeAnthropicAPI(_respond) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Kept out of the real allowlist: set by accident, it would bill the API.
        monkeypatch.setattr(cc, "_ENV_KEEP", (*cc._ENV_KEEP, "ANTHROPIC_API_KEY"))
        yield server


@pytest.fixture
def harness(workspace, tmp_path, monkeypatch, api):
    monkeypatch.setattr(cc, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(cc, "SKILLS_DIR", workspace / "skills")
    monkeypatch.setattr("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts")
    # No hushed here: the rewrite is marked instead, so the test sees it happen.
    monkeypatch.setattr(
        "aug.core.harness.claude_code_tools.hushed_command", lambda c: f"echo REWRITTEN; {c}"
    )

    def no_token(name: str) -> str:
        raise HushedError("not set")

    monkeypatch.setattr(cc, "read_secret", no_token)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text("# AGENTS\n\nI am AUG.\n")
    (workspace / "skills" / "weather").mkdir(parents=True)
    (workspace / "skills" / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Says what the weather is.\n---\nSunny.\n"
    )
    run_registry.clear()
    return _new_harness(tmp_path)


def _new_harness(tmp_path) -> ClaudeCodeHarness:
    profile = ClaudeCodeProfile(
        model="claude-opus-5-5",
        effort="high",
        builtin_tools=("Bash", "Read", "ToolSearch", "Skill"),
        tools=[echo],
    )
    return ClaudeCodeHarness(profile, config_dir=tmp_path / "claude")


async def _turn(harness: ClaudeCodeHarness, said: str) -> list:
    message = Message(parts=[TextPart(text=said)], sender=Sender(interface="rest_api", user_id="u"))
    async with run_registry.run_lock("ctx"):
        return [
            e async for e in harness.run(message, "ctx", TurnContext(reply_to=ADDR, attended=True))
        ]


def _reply(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta) and e.parent_id is None)


async def _in(harness: ClaudeCodeHarness, *turns: str) -> list[list]:
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        return [await _turn(harness, said) for said in turns]


async def test_a_plain_turn(harness, api):
    (events,) = await _in(harness, "hi")
    assert _reply(events) == "hello there"
    assert isinstance(events[-1], TurnEnded) and events[-1].error is None
    systems = [" ".join(b.get("text", "") for b in r.get("system", [])) for r in api.requests]
    assert any("I am AUG." in system for system in systems)


async def test_the_workspace_skills_are_listed_as_aug_skills(harness, api):
    await _in(harness, "hi")
    assert any("aug:weather" in json.dumps(r) for r in api.requests)


async def test_an_aug_tool_runs_in_process_with_the_turn_context(harness):
    (events,) = await _in(harness, "MARK:echo")
    outputs = [e.output for e in events if isinstance(e, ToolEnded) and e.name == "echo"]
    assert outputs == ["echo:x:ctx"]


async def test_bash_is_rewritten_and_never_sees_aug_environment(harness, monkeypatch):
    monkeypatch.setenv("API_KEY", "aug-secret")
    (events,) = await _in(harness, "MARK:bash")
    (bash,) = [e for e in events if isinstance(e, ToolEnded) and e.name == "Bash"]
    assert "REWRITTEN" in bash.output
    assert "API_KEY=unset" in bash.output


async def test_a_blocked_path_is_refused(harness):
    (events,) = await _in(harness, "MARK:secret")
    (read,) = [e for e in events if isinstance(e, ToolEnded) and e.name == "Read"]
    assert read.error and "not permitted" in read.output


async def test_a_new_harness_resumes_the_session(harness, tmp_path, api):
    await _in(harness, "first message")
    await _in(_new_harness(tmp_path), "second message")

    said = [
        b.get("text", "")
        for m in api.requests[-1]["messages"]
        if m["role"] == "user"
        for b in (m["content"] if isinstance(m["content"], list) else [])
    ]
    assert any("first message" in s for s in said)
