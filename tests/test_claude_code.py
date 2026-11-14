"""ClaudeCodeHarness against a fake SDK client: the CLI's messages are scripted.

What is tested here is AUG's side: turn boundaries (result, then idle), event mapping,
self-started turns, the session record and resume, restarts on a setup change,
eviction, injection, cancel and error kinds.  The real CLI is exercised by hand.
"""

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from langchain_core.tools import tool
from mcp.types import ListToolsRequest

from aug.core.harness import (
    Address,
    FileEmitted,
    FilePart,
    HarnessEvent,
    Message,
    Sender,
    StatusUpdate,
    TextDelta,
    TextPart,
    ToolEnded,
    ToolStarted,
    TurnContext,
    TurnEnded,
    TurnStarted,
)
from aug.core.harness.claude_code import harness as cc
from aug.core.harness.claude_code.harness import ClaudeCodeHarness, ClaudeCodeProfile
from aug.core.harness.claude_code.tools import (
    ToolCalls,
    _adapt,
    canonical_name,
    guard_tool_use,
    mcp_content,
)
from aug.core.harness.tool_context import ToolContext
from aug.core.run import run_registry
from aug.core.tools.context import current_tool_context
from aug.core.tools.run_bash import hushed_command
from aug.utils.hushed import HushedError

SESSION = "11111111-2222-3333-4444-555555555555"
ADDR = Address(interface="rest_api", conversation_id="chat", chat_id="chat")


@tool
async def echo(text: str) -> str:
    """Echo *text* back."""
    ctx = current_tool_context()
    return f"echo:{text}:{ctx.context_id if ctx else 'none'}"


class FakeClient:
    """Stands in for ClaudeSDKClient.  ``on_query`` scripts the CLI's answer."""

    instances: list["FakeClient"] = []
    on_query: Callable[["FakeClient", list[dict]], None] | None = None

    def __init__(self, options: Any) -> None:
        self.options = options
        self.queries: list[dict] = []
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.interrupted = False
        self.closed = False
        FakeClient.instances.append(self)

    async def connect(self) -> None:
        # Claude writes the session file once the session starts.
        projects = Path(self.options.env["CLAUDE_CONFIG_DIR"]) / "projects" / "-workspace"
        projects.mkdir(parents=True, exist_ok=True)
        (projects / f"{SESSION}.jsonl").touch()

    async def disconnect(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)

    async def query(self, prompt: AsyncIterator[dict]) -> None:
        messages = [m async for m in prompt]
        self.queries.extend(messages)
        if FakeClient.on_query is not None:
            FakeClient.on_query(self, messages)

    async def interrupt(self) -> None:
        self.interrupted = True
        self.send(_result(terminal_reason="aborted_tools"), _state("idle"))

    async def receive_messages(self) -> AsyncIterator[Any]:
        while (message := await self.inbox.get()) is not None:
            if isinstance(message, Exception):
                raise message
            yield message

    def send(self, *messages: Any) -> None:
        for message in messages:
            self.inbox.put_nowait(message)


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.instances = []
    FakeClient.on_query = None
    monkeypatch.setattr("aug.core.harness.claude_code.session.ClaudeSDKClient", FakeClient)
    return FakeClient


@pytest.fixture
def harness(workspace, tmp_path, monkeypatch, fake_client):
    monkeypatch.setattr(cc, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(cc, "SKILLS_DIR", workspace / "skills")
    monkeypatch.setattr("aug.core.transcript.TRANSCRIPTS_DIR", tmp_path / "transcripts")

    def no_token(name: str) -> str:
        raise HushedError("not set")

    monkeypatch.setattr(cc, "read_secret", no_token)
    (workspace / "skills" / "weather").mkdir(parents=True)
    (workspace / "skills" / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: The weather.\n---\nSunny.\n"
    )
    (workspace / "AGENTS.md").write_text("# AGENTS\n\nI am AUG.\n")
    run_registry.clear()
    return _harness(tmp_path)


def _harness(tmp_path: Path, **profile: Any) -> ClaudeCodeHarness:
    settings = {
        "model": "claude-opus-5-5",
        "effort": "high",
        "builtin_tools": ("Bash", "Read"),
        "tools": [echo],
    } | profile
    return ClaudeCodeHarness(ClaudeCodeProfile(**settings), config_dir=tmp_path / "claude")


def _message(text: str = "hi") -> Message:
    return Message(parts=[TextPart(text=text)], sender=Sender(interface="rest_api", user_id="u"))


async def _turn(
    harness: ClaudeCodeHarness, text: str = "hi", context_id: str = "ctx", **turn: Any
) -> list[HarnessEvent]:
    events = []
    async with run_registry.run_lock(context_id):
        async for event in harness.run(
            _message(text), context_id, TurnContext(reply_to=ADDR, attended=True, **turn)
        ):
            events.append(event)
    return events


def _state(state: str) -> SystemMessage:
    return SystemMessage(subtype="session_state_changed", data={"state": state})


def _init() -> SystemMessage:
    return SystemMessage(subtype="init", data={"session_id": SESSION})


def _stream(text: str, message_id: str = "m1") -> list[StreamEvent]:
    return [
        StreamEvent(
            uuid="u",
            session_id=SESSION,
            event={"type": "message_start", "message": {"id": message_id}},
        ),
        StreamEvent(
            uuid="u",
            session_id=SESSION,
            event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        ),
    ]


def _assistant(*blocks: Any, message_id: str = "m1", error: Any = None) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks), model="claude", message_id=message_id, error=error
    )


def _result(**fields: Any) -> ResultMessage:
    base = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": SESSION,
        "usage": {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 100},
    }
    return ResultMessage(**(base | fields))


def _reply(text: str) -> Callable[[FakeClient, list[dict]], None]:
    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(
            _state("running"),
            _init(),
            *_stream(text),
            _assistant(TextBlock(text=text)),
            _result(),
            _state("idle"),
        )

    return answer


# -- turns ---------------------------------------------------------------------------


async def test_a_turn_streams_text_and_ends_at_idle(harness, fake_client):
    fake_client.on_query = _reply("hello")
    events = await _turn(harness)

    assert isinstance(events[0], TurnStarted)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["hello"]  # not twice
    ended = events[-1]
    assert isinstance(ended, TurnEnded) and ended.error is None and not ended.cancelled
    assert ended.usage.cache_read_tokens == 100


async def test_the_message_is_sent_verbatim_with_the_time_first(harness, fake_client):
    fake_client.on_query = _reply("ok")
    await _turn(harness, "what's up")

    sent = fake_client.instances[0].queries[0]
    assert sent["client_composed"] is True
    blocks = sent["message"]["content"]
    assert blocks[0]["text"].startswith("[") and blocks[0]["text"].endswith("UTC]")
    assert blocks[1] == {"type": "text", "text": "what's up"}


async def test_an_image_goes_inline_with_its_path(harness, fake_client, tmp_path):
    fake_client.on_query = _reply("nice")
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG fake")
    message = Message(
        parts=[TextPart(text="look"), FilePart(path=image, media_type="image/png", filename="p")],
        sender=Sender(interface="rest_api", user_id="u"),
    )
    async with run_registry.run_lock("ctx"):
        async for _ in harness.run(message, "ctx", TurnContext(reply_to=ADDR, attended=True)):
            pass

    blocks = fake_client.instances[0].queries[0]["message"]["content"]
    assert blocks[2]["type"] == "image" and blocks[2]["source"]["media_type"] == "image/png"
    assert str(image) in blocks[3]["text"]


async def test_a_turn_does_not_end_at_a_result_before_idle(harness, fake_client):
    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(_state("running"), *_stream("one"), _assistant(TextBlock(text="one")))
        client.send(_result())  # the CLI goes on with a message that arrived meanwhile
        client.send(*_stream("two", "m2"), _assistant(TextBlock(text="two"), message_id="m2"))
        client.send(_result(), _state("idle"))

    fake_client.on_query = answer
    events = await _turn(harness)

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "onetwo"
    assert sum(isinstance(e, TurnEnded) for e in events) == 1


async def test_tool_calls_carry_canonical_names_and_their_parent(harness, fake_client):
    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(
            _state("running"),
            _assistant(ToolUseBlock(id="t1", name="mcp__aug__echo", input={"text": "x"})),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="echo:x")]),
            AssistantMessage(
                content=[ToolUseBlock(id="t2", name="Bash", input={"command": "ls"})],
                model="claude",
                parent_tool_use_id="agent-1",
            ),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t2", content="boom", is_error=True)],
                parent_tool_use_id="agent-1",
            ),
            _result(),
            _state("idle"),
        )

    fake_client.on_query = answer
    events = await _turn(harness)

    started = [e for e in events if isinstance(e, ToolStarted)]
    ended = [e for e in events if isinstance(e, ToolEnded)]
    assert [(e.name, e.parent_id) for e in started] == [("echo", None), ("Bash", "agent-1")]
    assert [(e.name, e.output, e.error) for e in ended] == [
        ("echo", "echo:x", False),
        ("Bash", "boom", True),
    ]


async def test_the_session_starts_with_aug_prompt_skills_and_no_filesystem_settings(
    harness, fake_client
):
    fake_client.on_query = _reply("ok")
    await _turn(harness)

    options = fake_client.instances[0].options
    assert options.system_prompt["preset"] == "claude_code"
    assert "I am AUG." in options.system_prompt["append"]
    assert "get_skill" not in options.system_prompt["append"]  # Claude lists skills itself
    assert options.setting_sources == []
    assert options.skills == ["aug:weather"]
    assert options.permission_mode == "dontAsk"
    assert options.allowed_tools == ["Bash", "Read", "mcp__aug_core", "mcp__aug"]
    assert options.tools == ["Bash", "Read"]
    assert options.session_id is not None and options.resume is None
    assert Path(options.cli_path).name == "cli.sh"
    plugin = Path(options.plugins[0]["path"])
    assert (plugin / ".claude-plugin" / "plugin.json").exists()
    assert sorted(p.name for p in plugin.iterdir()) == [".claude-plugin", "skills"]
    assert (plugin / "skills").resolve() == (Path(options.cwd) / "skills").resolve()


# -- the session record -------------------------------------------------------------


async def test_a_new_harness_resumes_the_session_with_the_prompt_it_started_with(
    harness, fake_client, tmp_path, workspace
):
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    first_prompt = fake_client.instances[0].options.system_prompt["append"]

    (workspace / "AGENTS.md").write_text("# AGENTS\n\nI am someone else now.\n")
    again = _harness(tmp_path)
    await _turn(again)

    options = fake_client.instances[1].options
    assert options.resume == SESSION and options.session_id is None
    assert options.system_prompt["append"] == first_prompt
    assert await again.session_prompt("ctx", "rest_api") == first_prompt


async def test_a_new_context_gets_the_workspace_as_it_is_now(harness, fake_client, workspace):
    fake_client.on_query = _reply("ok")
    await _turn(harness, context_id="old")
    (workspace / "AGENTS.md").write_text("# AGENTS\n\nI am someone else now.\n")
    await _turn(harness, context_id="new")

    assert "someone else" in fake_client.instances[1].options.system_prompt["append"]


async def test_a_setup_change_restarts_the_process_and_resumes(harness, fake_client, tmp_path):
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    await _turn(harness)
    assert len(fake_client.instances) == 1  # warm

    @tool
    def extra() -> str:
        """A new tool."""
        return "x"

    harness.profile = replace(harness.profile, tools=[echo, extra])
    await _turn(harness)

    assert len(fake_client.instances) == 2
    assert fake_client.instances[0].closed
    assert fake_client.instances[1].options.resume == SESSION


async def test_always_loaded_tools_get_their_own_server_claude_always_loads(harness, fake_client):
    @tool
    def extra() -> str:
        """Another tool."""
        return "x"

    harness.profile = replace(
        harness.profile, tools=[echo, extra], always_loaded=frozenset({"echo"})
    )
    fake_client.on_query = _reply("ok")
    await _turn(harness)

    servers = fake_client.instances[0].options.mcp_servers
    assert servers["aug_core"]["alwaysLoad"] is True
    assert "alwaysLoad" not in servers["aug"]
    assert await _tool_names(servers["aug_core"]) == ["echo"]
    assert await _tool_names(servers["aug"]) == ["extra"]


async def test_changing_the_always_loaded_tools_restarts_the_process(harness, fake_client):
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    harness.profile = replace(harness.profile, always_loaded=frozenset({"echo"}))
    await _turn(harness)
    assert len(fake_client.instances) == 2


def test_a_profile_cannot_always_load_a_tool_it_lacks():
    with pytest.raises(ValueError, match="gmail_send"):
        ClaudeCodeProfile(
            model="m",
            effort="high",
            builtin_tools=(),
            tools=[echo],
            always_loaded=frozenset({"echo", "gmail_send"}),
        )


async def test_a_session_whose_file_is_gone_starts_afresh(harness, fake_client, tmp_path):
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    for path in (tmp_path / "claude" / "projects").glob("*/*.jsonl"):
        path.unlink()

    await _turn(_harness(tmp_path))
    options = fake_client.instances[1].options
    assert options.resume is None and options.session_id not in (None, SESSION)


async def test_a_setup_change_waits_for_background_work(harness, fake_client):
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    fake_client.instances[0].send(_task_started("b1"))
    await asyncio.sleep(0.01)

    @tool
    def extra() -> str:
        """A new tool."""
        return "x"

    harness.profile = replace(harness.profile, tools=[echo, extra])
    await _turn(harness)
    assert len(fake_client.instances) == 1  # not restarted under the running task


async def test_compact_sends_the_slash_command(harness, fake_client):
    assert not await harness.compact("ctx")  # nothing to compact yet

    fake_client.on_query = _reply("ok")
    await _turn(harness)

    def compacted(client: FakeClient, messages: list[dict]) -> None:
        # A local command: answered without the CLI going through "running".
        client.send(SystemMessage(subtype="compact_boundary", data={}), _result())

    fake_client.on_query = compacted
    assert await harness.compact("ctx")
    sent = fake_client.instances[0].queries[-1]
    assert sent["message"]["content"] == [{"type": "text", "text": "/compact"}]
    assert "client_composed" not in sent


async def test_a_turn_that_goes_idle_without_a_result_still_ends(harness, fake_client):
    fake_client.on_query = lambda client, messages: client.send(_state("running"), _state("idle"))
    events = await _turn(harness)
    assert isinstance(events[-1], TurnEnded)


async def test_a_message_injected_before_the_process_is_up_reaches_the_turn(harness, fake_client):
    fake_client.on_query = lambda client, messages: client.send(_state("running"))
    message = _message("first")
    async with run_registry.run_lock("ctx"):
        stream = harness.run(message, "ctx", TurnContext(reply_to=ADDR, attended=True))
        assert isinstance(await anext(stream), TurnStarted)
        assert await harness.inject(_message("second"), "ctx")  # no process yet
        rest = asyncio.create_task(_collect(stream))
        for _ in range(50):
            if fake_client.instances and len(fake_client.instances[0].queries) == 2:
                break
            await asyncio.sleep(0.01)
        fake_client.instances[0].send(_result(), _state("idle"))
        await rest

    texts = [q["message"]["content"][1]["text"] for q in fake_client.instances[0].queries]
    assert texts == ["first", "second"]


# -- self-started turns -------------------------------------------------------------


async def test_a_turn_the_cli_starts_itself_goes_to_the_sink(harness, fake_client):
    delivered: list[tuple[Address, str, list[HarnessEvent]]] = []

    async def sink(reply_to: Address, context_id: str, events: AsyncIterator[HarnessEvent]):
        delivered.append((reply_to, context_id, [e async for e in events]))

    fake_client.on_query = _reply("started")
    async with harness.running(sink):
        await _turn(harness)
        fake_client.instances[0].send(
            _state("running"), *_stream("done: 42"), _assistant(TextBlock(text="done: 42"))
        )
        fake_client.instances[0].send(_result(), _state("idle"))
        for _ in range(50):
            if delivered:
                break
            await asyncio.sleep(0.01)

    reply_to, context_id, events = delivered[0]
    assert (reply_to, context_id) == (ADDR, "ctx")
    assert isinstance(events[0], TurnStarted) and isinstance(events[-1], TurnEnded)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["done: 42"]
    assert run_registry.get("ctx") is None  # it registered no run of its own


async def test_a_user_turn_takes_over_a_self_started_one_and_nothing_empty_is_sent(
    harness, fake_client
):
    delivered = []

    async def sink(reply_to, context_id, events):
        delivered.append([e async for e in events])

    fake_client.on_query = _reply("ok")
    async with harness.running(sink):
        await _turn(harness)
        fake_client.instances[0].send(_state("running"))  # background work wakes the CLI
        await asyncio.sleep(0.01)
        fake_client.on_query = lambda client, messages: client.send(
            *_stream("for you"), _assistant(TextBlock(text="for you")), _result(), _state("idle")
        )
        events = await _turn(harness)
        await asyncio.sleep(0.05)

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["for you"]
    assert delivered == []


async def test_stop_reaches_a_self_started_turn(harness, fake_client):
    async def sink(reply_to, context_id, events):
        async for _ in events:
            pass

    fake_client.on_query = _reply("ok")
    async with harness.running(sink):
        await _turn(harness)
        client = fake_client.instances[0]
        client.send(_state("running"), *_stream("working"))
        await asyncio.sleep(0.01)
        assert await harness.cancel("ctx")
        assert client.interrupted


def _task_started(task_id: str) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id=task_id,
        description="sleep",
        uuid="u",
        session_id=SESSION,
    )


async def _collect(stream: AsyncIterator[HarnessEvent]) -> list[HarnessEvent]:
    return [e async for e in stream]


async def test_a_process_with_background_work_is_never_evicted(harness, fake_client):
    harness._idle_seconds = 0
    fake_client.on_query = _reply("ok")
    await _turn(harness)
    client = fake_client.instances[0]
    client.send(_task_started("b1"))
    await asyncio.sleep(0.01)
    await harness._evict()
    assert not client.closed

    client.send(
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="b1",
            status="completed",
            output_file="",
            summary="",
            uuid="u",
            session_id=SESSION,
        )
    )
    await asyncio.sleep(0.01)
    await harness._evict()
    assert client.closed


async def test_only_the_idle_cap_is_kept_warm(harness, fake_client):
    harness._max_idle = 1
    fake_client.on_query = _reply("ok")
    await _turn(harness, context_id="a")
    await _turn(harness, context_id="b")
    await harness._evict()

    assert [c.closed for c in fake_client.instances] == [True, False]


# -- inject, cancel, errors ---------------------------------------------------------


async def test_inject_reaches_the_turn_in_flight_only(harness, fake_client):
    assert not await harness.inject(_message("late"), "ctx")

    def answer(client: FakeClient, messages: list[dict]) -> None:
        if messages[0]["message"]["content"][1]["text"] == "first":
            client.send(_state("running"))  # still working when the second message lands

    fake_client.on_query = answer
    turn = asyncio.create_task(_turn(harness, "first"))
    for _ in range(50):
        if fake_client.instances and fake_client.instances[0].queries:
            break
        await asyncio.sleep(0.01)
    assert await harness.inject(_message("second"), "ctx")
    fake_client.instances[0].send(_result(), _state("idle"))
    await turn

    texts = [q["message"]["content"][1]["text"] for q in fake_client.instances[0].queries]
    assert texts == ["first", "second"]


async def test_cancel_interrupts_and_ends_cancelled(harness, fake_client):
    fake_client.on_query = lambda client, messages: client.send(_state("running"))
    turn = asyncio.create_task(_turn(harness))
    for _ in range(50):
        if fake_client.instances and fake_client.instances[0].queries:
            break
        await asyncio.sleep(0.01)

    assert await harness.cancel("ctx")
    events = await turn
    assert fake_client.instances[0].interrupted
    assert events[-1].cancelled


@pytest.mark.parametrize(
    "error, kind",
    [
        ("authentication_failed", "login"),
        ("billing_error", "auth"),
        ("rate_limit", "rate_limited"),
        ("server_error", "unavailable"),
        ("unknown", "internal"),
    ],
)
async def test_assistant_errors_map_to_kinds(harness, fake_client, error, kind):
    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(
            _state("running"),
            _assistant(TextBlock(text="API Error"), error=error),
            _result(is_error=True),
            _state("idle"),
        )

    fake_client.on_query = answer
    events = await _turn(harness)
    assert events[-1].error.kind == kind


async def test_a_rejected_rate_limit_says_when_it_resets(harness, fake_client):
    resets = int(datetime(2026, 11, 14, 14, 0, tzinfo=UTC).timestamp())

    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(
            _state("running"),
            RateLimitEvent(
                rate_limit_info=RateLimitInfo(status="rejected", resets_at=resets),
                uuid="u",
                session_id=SESSION,
            ),
            _result(is_error=True, api_error_status=429),
            _state("idle"),
        )

    fake_client.on_query = answer
    error = (await _turn(harness))[-1].error
    assert error.kind == "rate_limited" and error.retry_at.hour == 14


async def test_a_rate_limit_warning_is_a_status_once(harness, fake_client):
    warning = RateLimitEvent(
        rate_limit_info=RateLimitInfo(
            status="allowed_warning", utilization=0.9, rate_limit_type="five_hour"
        ),
        uuid="u",
        session_id=SESSION,
    )

    def answer(client: FakeClient, messages: list[dict]) -> None:
        client.send(_state("running"), warning, warning, _result(), _state("idle"))

    fake_client.on_query = answer
    statuses = [e.text for e in await _turn(harness) if isinstance(e, StatusUpdate)]
    assert statuses == ["Claude usage is at 90% of the five hour limit."]


async def test_a_dead_process_ends_the_turn_and_is_reopened(harness, fake_client):
    fake_client.on_query = lambda client, messages: client.send(
        _state("running"), RuntimeError("CLI exited")
    )
    events = await _turn(harness)
    assert events[-1].error.kind == "unavailable"

    fake_client.on_query = _reply("back")
    await _turn(harness)
    assert len(fake_client.instances) == 2


# -- tools ----------------------------------------------------------------------------


def test_canonical_names_drop_the_aug_server_prefix():
    assert canonical_name("mcp__aug__run_ssh") == "run_ssh"
    assert canonical_name("mcp__aug_core__note") == "note"
    assert canonical_name("playwright__browser_click") == "playwright__browser_click"
    assert canonical_name("Bash") == "Bash"


async def test_the_adapter_runs_the_tool_with_the_turn_context_and_files_events_under_it():
    calls = ToolCalls()
    emitted = []

    async def collect(event):
        emitted.append(event)

    calls.context = ToolContext(
        context_id="ctx", turn=TurnContext(reply_to=ADDR, attended=True), _emit=collect
    )

    @tool
    async def sender(name: str) -> str:
        """Send a file."""
        ctx = current_tool_context()
        await ctx.emit(FileEmitted(file_id="f", path=Path("/x"), media_type="a/b", filename=name))
        return "sent"

    calls.announce("toolu_1", "mcp__aug_core__sender", {"name": "x"})
    result = await _adapt(sender, calls).handler({"name": "x"})

    assert result == {"content": [{"type": "text", "text": "sent"}], "is_error": False}
    assert emitted[0].parent_id == "toolu_1"


async def test_a_tool_that_raises_is_an_error_result():
    @tool
    async def broken() -> str:
        """Fails."""
        raise RuntimeError("nope")

    result = await _adapt(broken, ToolCalls()).handler({})
    assert result["is_error"] is True and "nope" in result["content"][0]["text"]


def test_tool_images_become_mcp_image_blocks():
    assert mcp_content("") == [{"type": "text", "text": "(no output)"}]
    assert mcp_content(
        [
            {"type": "text", "text": "shot"},
            {"type": "image", "base64": "AAA", "mime_type": "image/png"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
        ]
    ) == [
        {"type": "text", "text": "shot"},
        {"type": "image", "data": "AAA", "mimeType": "image/png"},
        {"type": "image", "data": "BBB", "mimeType": "image/jpeg"},
    ]


async def test_bash_runs_through_hushed():
    out = await guard_tool_use(
        {"tool_name": "Bash", "tool_input": {"command": "ls -la", "description": "list"}},
        "t",
        {"signal": None},
    )
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated == {"command": hushed_command("ls -la"), "description": "list"}
    assert updated["command"].startswith("hushed run -- bash -c")


async def test_monitor_commands_are_checked_not_rewritten():
    blocked = await guard_tool_use(
        {"tool_name": "Monitor", "tool_input": {"command": "tail -f /app/data/settings.json"}},
        "t",
        {"signal": None},
    )
    assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"
    allowed = await guard_tool_use(
        {"tool_name": "Monitor", "tool_input": {"command": "tail -f /tmp/log"}},
        "t",
        {"signal": None},
    )
    assert allowed == {}


def test_hushed_finds_its_store_from_the_cli():
    assert "HUSHED_DIR" in cc._ENV_KEEP


@pytest.mark.parametrize(
    "tool_name, tool_input",
    [
        ("Bash", {"command": "cat /app/data/.hushed/key"}),
        ("Read", {"file_path": "/app/data/settings.json"}),
        ("Write", {"file_path": "/app/data/keys/id_rsa", "content": "x"}),
    ],
)
async def test_blocked_paths_are_refused(tool_name, tool_input):
    out = await guard_tool_use(
        {"tool_name": tool_name, "tool_input": tool_input}, "t", {"signal": None}
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


async def test_other_files_pass_untouched():
    out = await guard_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": "/tmp/notes.txt"}}, "t", {"signal": None}
    )
    assert out == {}


def test_the_cli_gets_only_the_allowlisted_environment():
    env = {
        **os.environ,
        "AUG_CLAUDE_CLI": "/usr/bin/env",
        "AUG_CLAUDE_ENV_KEEP": " ".join(cc._ENV_KEEP),
        "CLAUDE_CONFIG_DIR": "/app/data/claude",
        "DATABASE_URL": "postgresql://secret",
        "API_KEY": "secret",
    }
    out = subprocess.run(
        [str(cc._CLI_WRAPPER)], env=env, capture_output=True, text=True, check=True
    ).stdout
    names = {line.split("=", 1)[0] for line in out.splitlines()}
    assert "CLAUDE_CONFIG_DIR" in names
    assert names <= set(cc._ENV_KEEP)


async def _tool_names(server: Any) -> list[str]:
    handler = server["instance"].request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest(method="tools/list"))
    return [t.name for t in result.root.tools]
