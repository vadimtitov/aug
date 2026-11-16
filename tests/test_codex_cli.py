"""CodexHarness with the real Codex app-server against a fake Responses API.

Free: no ChatGPT account is used.  Covers what scripted notifications can't: the
JSON-RPC client, the guard hook and its trust, AUG's tools as dynamic tools, the
environment cut, steering, stopping, compaction, oneshot, and resume from Codex's own
thread files.
"""

import asyncio
import json
import sys
from typing import Any

import codex_cli_bin
import pytest
from langchain_core.tools import tool

from aug.core.harness import (
    Address,
    Message,
    Sender,
    TextDelta,
    TextPart,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnContext,
    TurnEnded,
    TurnStarted,
)
from aug.core.harness.base import error_text
from aug.core.harness.codex import harness as codex_harness
from aug.core.harness.codex import runtime as codex_runtime
from aug.core.harness.codex.harness import CodexHarness, CodexProfile
from aug.core.harness.codex.runtime import CodexRuntime
from aug.core.harness.codex.tools import AUG_NAMESPACE
from aug.core.run import run_registry
from aug.core.tools.context import current_tool_context
from tests.codex_fake_api import (
    FAKE_MODEL,
    ApiError,
    FakeResponsesAPI,
    call,
    codex_provider_config,
    developer_text,
    last_input,
    message,
    said,
)

try:
    codex_cli_bin.bundled_codex_path()
    _HAS_CODEX = True
except FileNotFoundError:
    _HAS_CODEX = False

pytestmark = pytest.mark.skipif(not _HAS_CODEX, reason="no bundled Codex binary")

ADDR = Address(interface="rest_api", conversation_id="chat", chat_id="chat")


@tool
async def echo(value: str) -> str:
    """Echo *value* back, with the chat it ran for."""
    ctx = current_tool_context()
    return f"echo:{value}:{ctx.context_id if ctx else 'none'}"


@tool
async def stamp() -> str:
    """Stamp the chat."""
    return "stamped"


@tool
async def slow() -> str:
    """Take a long time."""
    ctx = current_tool_context()
    assert ctx is not None
    await ctx.emit(ToolProgress(text="started"))
    await asyncio.sleep(30)
    return "slow done"


def _respond(request: dict[str, Any]) -> list[dict[str, Any]] | ApiError:
    """The scripted model: a marker in the user's latest text picks what it does; a
    tool result is reported back as text."""
    last = last_input(request)
    if last.get("type") == "function_call_output":
        output = last.get("output")
        if isinstance(output, list):
            output = " ".join(str(i.get("text", "")) for i in output if isinstance(i, dict))
        return [message(f"result: {output}")]
    text = json.dumps(last)
    if "MARK:stamp" in text:
        return [call("c-stamp", "stamp", {})]
    if "MARK:echo" in text:
        return [call("c-echo", "echo", {"value": "x"}, namespace=AUG_NAMESPACE)]
    if "MARK:slow" in text:
        return [call("c-slow", "slow", {})]
    if "MARK:home" in text:
        return [call("c-home", "exec_command", {"cmd": "echo HOME_IS=${CODEX_HOME:-unset}"})]
    if "MARK:shell" in text:
        return [call("c-shell", "exec_command", {"cmd": "echo API_KEY=${API_KEY:-unset}"})]
    if "MARK:secret" in text:
        return [call("c-secret", "exec_command", {"cmd": "cat /app/data/settings.json"})]
    if "MARK:logged-out" in text:
        return ApiError(401, "Your authentication token has been invalidated.")
    return [message("hello there")]


@pytest.fixture
def api(monkeypatch):
    with FakeResponsesAPI(_respond) as server:
        yield server


@pytest.fixture
def harness(workspace, tmp_path, monkeypatch, api, fake_hushed):
    monkeypatch.setattr(codex_runtime, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(codex_runtime, "SKILLS_DIR", workspace / "skills")
    monkeypatch.setattr(codex_harness, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setenv("API_KEY", "aug-secret")  # AUG's own: must never reach Codex
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no skills of this machine's user
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text("# AGENTS\n\nI am AUG.\n")
    (workspace / "skills" / "weather").mkdir(parents=True)
    (workspace / "skills" / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Says what the weather is.\n---\nSunny.\n"
    )
    run_registry.clear()
    return _new_harness(tmp_path, api)


def _new_harness(tmp_path, api: FakeResponsesAPI) -> CodexHarness:
    runtime = CodexRuntime(
        tmp_path / "codex",
        config=codex_provider_config(api.url),
        env={"FAKE_API_KEY": "test-key"},
    )
    profile = CodexProfile(
        model=FAKE_MODEL,
        effort="low",
        tools=[echo, stamp, slow],
        always_loaded=frozenset({"stamp", "slow"}),
    )
    return CodexHarness(profile, runtime=runtime)


def _message(text: str) -> Message:
    return Message(parts=[TextPart(text=text)], sender=Sender(interface="rest_api", user_id="u"))


async def _turn(harness: CodexHarness, text: str) -> list:
    async with run_registry.run_lock("ctx"):
        return [
            e
            async for e in harness.run(
                _message(text), "ctx", TurnContext(reply_to=ADDR, attended=True)
            )
        ]


async def _slow_turn(harness: CodexHarness) -> asyncio.Task[list]:
    """A turn running the slow tool, returned once the tool has started."""
    started = asyncio.Event()

    async def run() -> list:
        async with run_registry.run_lock("ctx"):
            turn = TurnContext(reply_to=ADDR, attended=True)
            events = []
            async for event in harness.run(_message("MARK:slow"), "ctx", turn):
                events.append(event)
                if isinstance(event, ToolProgress):
                    started.set()
            return events

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 30)
    return task


def _reply(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta) and e.parent_id is None)


async def _in(harness: CodexHarness, *turns: str) -> list[list]:
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        return [await _turn(harness, text) for text in turns]


def _calls(api: FakeResponsesAPI) -> list[dict[str, Any]]:
    """The model calls of chat threads (not a oneshot's)."""
    return [r for r in api.requests if r.get("model") == FAKE_MODEL]


async def test_a_plain_turn_carries_aug_prompt_and_skills(harness, api):
    (events,) = await _in(harness, "hi")
    assert _reply(events) == "hello there"
    ended = events[-1]
    assert isinstance(ended, TurnEnded) and ended.error is None
    assert ended.context is not None and ended.context.window > 0 and ended.context.used > 0
    assert ended.usage.input_tokens == 800 and ended.usage.cache_read_tokens == 200
    developer = developer_text(_calls(api)[-1])
    assert "I am AUG." in developer
    assert "weather" in developer  # the workspace's skills, as Codex skills
    assert "skill-creator" not in developer  # Codex's bundled ones are off


async def test_an_always_loaded_tool_runs_with_the_turn_context(harness, api):
    (events,) = await _in(harness, "MARK:stamp")
    (started,) = [e for e in events if isinstance(e, ToolStarted)]
    (ended,) = [e for e in events if isinstance(e, ToolEnded)]
    assert (started.name, ended.name, ended.output, ended.error) == (
        "stamp",
        "stamp",
        "stamped",
        False,
    )
    tools = {t.get("name") for t in _calls(api)[-1].get("tools") or []}
    assert "stamp" in tools and "echo" not in tools  # echo waits in the aug namespace
    assert AUG_NAMESPACE in developer_text(_calls(api)[-1])


async def test_a_deferred_tool_runs_in_process(harness):
    (events,) = await _in(harness, "MARK:echo")
    outputs = [e.output for e in events if isinstance(e, ToolEnded) and e.name == "echo"]
    assert outputs == ["echo:x:ctx"]


async def test_a_shell_command_runs_through_hushed_without_aug_environment(harness, fake_hushed):
    (events,) = await _in(harness, "MARK:shell")
    (started,) = [e for e in events if isinstance(e, ToolStarted)]
    (ended,) = [e for e in events if isinstance(e, ToolEnded)]
    # Shown as the agent wrote it, run as the guard rewrote it.
    assert started.name == "Bash"
    assert started.args == {"command": "echo API_KEY=${API_KEY:-unset}"}
    assert "API_KEY=${API_KEY:-unset}" in fake_hushed.read_text()
    assert "API_KEY=unset" in ended.output and "aug-secret" not in ended.output


async def test_a_blocked_path_is_refused(harness, api, fake_hushed):
    await _in(harness, "MARK:secret")
    assert "not permitted" in json.dumps(last_input(_calls(api)[-1]))
    assert not fake_hushed.exists()  # nothing ran


async def test_a_guard_that_cannot_run_refuses_the_call(harness, api, fake_hushed, monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")  # the hook can't start
    await _in(harness, "MARK:shell")
    assert "could not check" in json.dumps(last_input(_calls(api)[-1]))
    assert not fake_hushed.exists()


async def test_a_new_harness_resumes_the_thread(harness, tmp_path, api):
    await _in(harness, "first message")
    await _in(_new_harness(tmp_path, api), "second message")
    first, last = _calls(api)[0], _calls(api)[-1]
    assert "first message" in said(last)
    # The same prompt, once, and the same prefix: the prompt cache holds across restarts.
    assert developer_text(last).count("I am AUG.") == 1
    assert last["input"][: len(first["input"])] == first["input"]


async def test_a_message_sent_mid_turn_joins_it(harness, api):
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        turn = await _slow_turn(harness)
        assert await harness.inject(_message("also this"), "ctx")
        await harness.cancel("ctx")
        events = await turn
    assert isinstance(events[-1], TurnEnded) and events[-1].cancelled


async def test_stop_ends_the_turn_and_its_tool(harness):
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        turn = await _slow_turn(harness)
        assert await harness.cancel("ctx")
        events = await asyncio.wait_for(turn, 20)
        ended = events[-1]
        assert isinstance(ended, TurnEnded) and ended.cancelled and ended.error is None
        # The thread takes the next turn.
        after = await _turn(harness, "hi")
    assert _reply(after) == "hello there"


async def test_compact_summarises_the_thread(harness):
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        assert not await harness.compact("ctx")  # nothing to compact yet
        await _turn(harness, "hi")
        assert await harness.compact("ctx")


async def test_oneshot_is_a_bare_call(harness, api):
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        assert await harness.oneshot("say hi", system="Be brief.") == "hello there"
    request = api.requests[-1]
    tools = {t.get("name") or t.get("type") for t in request.get("tools") or []}
    assert tools <= {"request_user_input"}  # Plan mode's only: never offered here
    assert "Be brief." in developer_text(request) or "Be brief." in str(request.get("instructions"))


async def test_a_rejected_login_tells_the_user_to_sign_in(harness):
    (events,) = await _in(harness, "MARK:logged-out")
    ended = events[-1]
    assert isinstance(ended, TurnEnded) and ended.error is not None
    assert ended.error.kind == "login"
    assert "/codex_login" in error_text(ended.error)


async def test_signed_out_codex_asks_for_a_login_without_a_model_call(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.setattr(codex_runtime, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(codex_runtime, "SKILLS_DIR", workspace / "skills")
    run_registry.clear()
    profile = CodexProfile(model=FAKE_MODEL, effort="low", tools=[])
    harness = CodexHarness(profile, runtime=CodexRuntime(tmp_path / "codex"))
    (events,) = await _in(harness, "hi")
    assert isinstance(events[0], TurnStarted)
    ended = events[-1]
    assert isinstance(ended, TurnEnded) and ended.error is not None
    assert "/codex_login" in error_text(ended.error)


async def test_status_reports_the_context_without_starting_a_thread(harness):
    async def sink(*args: Any) -> None:
        pass

    async with harness.running(sink):
        before = await harness.session_status("ctx")
        await _turn(harness, "hi")
        after = await harness.session_status("ctx")
    assert before.context is None and before.model == FAKE_MODEL
    assert after.context is not None and after.context.used > 0


async def test_commands_never_learn_where_the_login_is(harness):
    (events,) = await _in(harness, "MARK:home")
    (ended,) = [e for e in events if isinstance(e, ToolEnded)]
    assert "HOME_IS=unset" in ended.output
