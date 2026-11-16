"""invoke_tool: an AUG tool run for an agent CLI, its result as text and images."""

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from aug.core.harness.tool_bridge import ToolImage, ToolOutcome, invoke_tool, result_blocks
from aug.core.harness.tool_context import ToolContext
from aug.core.harness.types import Address, StatusUpdate, TurnContext
from aug.core.tools.context import current_tool_context

ADDR = Address(interface="rest_api", conversation_id="chat", chat_id="chat")


@tool
async def whoami() -> str:
    """Say which chat it runs for, and report progress."""
    ctx = current_tool_context()
    assert ctx is not None
    await ctx.emit(StatusUpdate(text="working"))
    return f"for:{ctx.context_id}"


@tool
async def broken() -> str:
    """Fails."""
    raise RuntimeError("nope")


@tool
async def ping() -> str:
    """Takes no arguments."""
    return "pong"


class _Refusing(BaseTool):
    """A tool whose answer is a ToolMessage with an error status."""

    name: str = "refusing"
    description: str = "Refuses."

    def _run(self) -> str:
        raise NotImplementedError

    async def _arun(self) -> str:
        raise NotImplementedError

    async def ainvoke(self, input, config=None, **kwargs):
        return ToolMessage(content="no", tool_call_id=input["id"], status="error")


async def test_a_tool_runs_with_the_turn_context():
    events = []

    async def emit(event):
        events.append(event)

    ctx = ToolContext(context_id="ctx", turn=TurnContext(reply_to=ADDR, attended=True), _emit=emit)
    assert await invoke_tool(whoami, {}, "call-1", ctx) == ToolOutcome(blocks=("for:ctx",))
    assert [e.text for e in events] == ["working"]


async def test_a_tool_that_raises_is_a_failed_outcome():
    outcome = await invoke_tool(broken, {}, "call-1", None)
    assert outcome.error and "nope" in outcome.blocks[0]


async def test_arguments_that_are_not_an_object_are_taken_as_none():
    assert await invoke_tool(ping, "garbage", None, None) == ToolOutcome(blocks=("pong",))


async def test_a_tool_message_with_an_error_status_is_a_failure():
    outcome = await invoke_tool(_Refusing(), {}, "c", None)
    assert outcome == ToolOutcome(blocks=("no",), error=True)


def test_results_become_text_and_images():
    assert result_blocks("") == ("(no output)",)
    assert result_blocks([]) == ("(no output)",)
    assert result_blocks(42) == ("42",)
    assert result_blocks(
        [
            "plain",
            {"type": "text", "text": "shot"},
            {"type": "image", "base64": "AAA", "mime_type": "image/png"},
            {"type": "image", "data": b"\x00", "mimeType": "image/gif"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]
    ) == (
        "plain",
        "shot",
        ToolImage(media_type="image/png", data="AAA"),
        ToolImage(media_type="image/gif", data="AA=="),
        ToolImage(media_type="image/jpeg", data="BBB"),
        '{"type": "image_url", "image_url": {"url": "https://x/y.png"}}',
    )
