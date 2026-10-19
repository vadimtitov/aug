"""KB tools, the KB agent's per-turn binding, and the REST pieces the KB relies on."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from aug.api.interfaces.base import FileContent, TextContent
from aug.api.interfaces.rest import RestApiInterface, _RestContext
from aug.api.schemas.chat import Attachment, ChatRequest
from aug.api.schemas.files import FileMetadata
from aug.core.agents.kb_agent import KbAugAgent
from aug.core.kb.models import QueryResult, WriteResult
from aug.core.kb.query import UnknownPredicateError
from aug.core.state import AgentState
from aug.core.tools.context import current_user_id
from aug.core.tools.kb import KB_TOOLS, kb_query, kb_tools_for_turn, kb_write
from aug.utils.kb.store import EntityRow, PredicateRow


def _config(user_id: str = "owner-1") -> dict:
    return {"configurable": {"thread_id": "t1", "interface": "rest_api", "user_id": user_id}}


def _predicate(name: str) -> PredicateRow:
    return PredicateRow(
        name=name, description="", value_type="literal", single_valued=True, use_count=1
    )


def _fake_kb(predicates=("phone", "home_address"), entities=()) -> MagicMock:
    kb = MagicMock()
    kb.predicates = AsyncMock(return_value=[_predicate(p) for p in predicates])
    kb.entities = AsyncMock(return_value=list(entities))
    kb.write = AsyncMock(return_value=WriteResult("obj:1", "note", ["me x 1"], [], "ok"))
    kb.query = AsyncMock(return_value=QueryResult([], 0, 0))
    return kb


# ── tools ───────────────────────────────────────────────────────────────────


async def test_tools_report_unavailable_kb_plainly():
    with patch("aug.core.tools.kb.get_knowledge_base", return_value=None):
        out = await kb_write.ainvoke({"knowledge": "x"}, config=_config())
    assert "unavailable" in out and "NOTHING was saved" in out


async def test_kb_write_uses_the_runs_owner_and_source():
    kb = _fake_kb()
    with patch("aug.core.tools.kb.get_knowledge_base", return_value=kb):
        out = await kb_write.ainvoke({"knowledge": "x"}, config=_config("u-42"))
    assert json.loads(out)["status"] == "ok"
    args = kb.write.await_args
    assert args.args[0] == "u-42" and args.kwargs["source"] == "api"


async def test_kb_query_unknown_predicate_is_an_error_string():
    kb = _fake_kb()
    kb.query = AsyncMock(side_effect=UnknownPredicateError("Unknown predicate 'x'"))
    with patch("aug.core.tools.kb.get_knowledge_base", return_value=kb):
        out = await kb_query.ainvoke(
            {"filters": [{"type": "predicate", "predicate": "x"}]}, config=_config()
        )
    assert out.startswith("Error:") and "Unknown predicate" in out


async def test_tools_for_turn_narrow_predicate_to_enum():
    tools = await kb_tools_for_turn(_fake_kb(), "owner-1")
    schema = json.dumps(convert_to_openai_tool(next(t for t in tools if t.name == "kb_query")))
    assert '"enum": ["phone", "home_address"]' in schema
    assert [t.name for t in tools] == [t.name for t in KB_TOOLS]
    assert kb_query.args_schema.model_fields["filters"]  # the shared tool is untouched


async def test_tools_for_turn_drop_predicate_filter_when_vocabulary_empty():
    tools = await kb_tools_for_turn(_fake_kb(predicates=()), "owner-1")
    schema = json.dumps(convert_to_openai_tool(next(t for t in tools if t.name == "kb_query")))
    assert '"predicate"' not in schema and '"date"' in schema


async def test_tools_for_turn_fall_back_when_kb_errors():
    kb = _fake_kb()
    kb.predicates = AsyncMock(side_effect=ConnectionError("down"))
    assert await kb_tools_for_turn(kb, "owner-1") == KB_TOOLS


# ── agent ───────────────────────────────────────────────────────────────────


def _agent() -> KbAugAgent:
    with patch("aug.core.agents.chat_agent.build_chat_model") as build:
        build.return_value = MagicMock()
        agent = KbAugAgent(model="m", tools=[kb_write])  # duplicates are dropped
    return agent


def test_agent_carries_each_kb_tool_once():
    assert [t.name for t in _agent().tools].count("kb_write") == 1


async def test_agent_rebinds_narrowed_tools_every_turn():
    agent = _agent()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
    agent._model.bind_tools = MagicMock(return_value=bound)
    state = AgentState(messages=[HumanMessage(content="hello")])
    with patch("aug.core.agents.kb_agent.get_knowledge_base", return_value=_fake_kb()):
        await agent.respond(state, _config())
    tools = agent._model.bind_tools.call_args.args[0]
    query_tool = next(t for t in tools if t.name == "kb_query")
    assert '"phone"' in json.dumps(convert_to_openai_tool(query_tool))


@tool
def _other_tool() -> str:
    """Any non-KB tool."""
    return ""


async def test_agent_drops_kb_tools_once_the_turn_hits_the_call_cap():
    with patch("aug.core.agents.chat_agent.build_chat_model") as build:
        build.return_value = MagicMock()
        agent = KbAugAgent(model="m", tools=[_other_tool])
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    agent._model.bind_tools = MagicMock(return_value=bound)
    loop = []
    for i in range(15):
        call = {"name": "kb_relate", "args": {}, "id": f"c{i}", "type": "tool_call"}
        loop += [AIMessage(content="", tool_calls=[call]), ToolMessage("{}", tool_call_id=f"c{i}")]
    earlier_turn = [HumanMessage(content="old"), *loop]
    kb = _fake_kb()
    with patch("aug.core.agents.kb_agent.get_knowledge_base", return_value=kb):
        # The cap counts only the current turn: an earlier busy turn doesn't count.
        await agent.respond(AgentState(messages=[*earlier_turn, HumanMessage("new")]), _config())
        names = [t.name for t in agent._model.bind_tools.call_args.args[0]]
        assert "kb_relate" in names

        await agent.respond(AgentState(messages=[HumanMessage(content="m"), *loop]), _config())
    names = [t.name for t in agent._model.bind_tools.call_args.args[0]]
    assert not {t.name for t in KB_TOOLS} & set(names)
    assert names == ["_other_tool"]  # the rest of the toolset is untouched


async def test_agent_prompt_has_kb_section_and_entity_index():
    agent = _agent()
    anna = EntityRow(
        "person:anna-logunova", "person", "Anna Logunova", [], datetime.now(tz=UTC), None
    )
    state = AgentState(messages=[HumanMessage(content="hi")], interface="rest_api")
    with patch(
        "aug.core.agents.kb_agent.get_knowledge_base", return_value=_fake_kb(entities=[anna])
    ):
        update = await agent.preprocess(state, _config())
    assert "<knowledge_base>" in update.system_prompt
    assert "Anna Logunova [person:anna-logunova]" in update.system_prompt


# ── REST ────────────────────────────────────────────────────────────────────


async def _incoming(request: ChatRequest, storage=None):
    interface = RestApiInterface(MagicMock(), storage)
    return await interface.receive_message(_RestContext(request=request, agent_version="v13"))


async def test_rest_user_defaults_to_thread_and_can_be_set():
    assert (await _incoming(ChatRequest(thread_id="t1", message="m"))).user_id == "t1"
    explicit = ChatRequest(thread_id="t1", message="m", user_id="max")
    assert (await _incoming(explicit)).user_id == "max"


async def test_rest_attachments_become_file_parts(tmp_path: Path):
    storage = MagicMock()
    storage.read = AsyncMock(side_effect=lambda fid: b"%PDF" if fid == "f-123456789" else None)
    storage.get_metadata = AsyncMock(
        return_value=FileMetadata(
            file_id="f-123456789",
            filename="ticket.pdf",
            size_bytes=4,
            created_at=datetime.now(tz=UTC),
        )
    )
    request = ChatRequest(
        thread_id="t1",
        message="my ticket",
        attachments=[Attachment(file_id="f-123456789"), Attachment(file_id="missing")],
    )
    with patch("aug.api.interfaces.rest.UPLOADS_DIR", tmp_path):
        incoming = await _incoming(request, storage)
    file_part, missing, text = incoming.parts
    assert isinstance(file_part, FileContent) and file_part.mime_type == "application/pdf"
    assert Path(file_part.path).read_bytes() == b"%PDF"
    assert Path(file_part.path).name.endswith("ticket.pdf")
    assert isinstance(missing, TextContent) and "could not be found" in missing.text
    assert text.text == "my ticket"


@pytest.mark.parametrize(
    ("configurable", "expected"),
    [
        ({"user_id": "u", "sender_id": "s"}, "u"),
        ({"sender_id": "s"}, "s"),
        ({}, ""),
    ],
)
def test_current_user_id_falls_back_to_sender(configurable, expected):
    with patch("aug.core.tools.context.get_config", return_value={"configurable": configurable}):
        assert current_user_id() == expected
