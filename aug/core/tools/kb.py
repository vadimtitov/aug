"""Knowledge base tools — kb_write, kb_relate, kb_query, kb_get, kb_forget.

Thin wrappers: the owner comes from the run (never from the model), the work is done
by ``KnowledgeBase``, and every call logs one structured ``kb_tool`` line.  Results
are JSON so the model reads exact fields; failures say plainly what did not happen.

``kb_tools_for_turn()`` returns copies whose kb_query schema narrows
``PredicateFilter.predicate`` to the owner's real predicates — the agent rebinds
them every turn as the vocabulary grows.
"""

import dataclasses
import json
import logging
import time
from datetime import date
from typing import Any, Literal, Union

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, ValidationError, create_model

from aug.core.kb.ingest import KbError
from aug.core.kb.models import (
    DateFilter,
    KbForgetInput,
    KbGetInput,
    KbQueryInput,
    KbRelateInput,
    KbWriteInput,
    KindFilter,
    LiteralValue,
    NodeValue,
    PredicateFilter,
    SourceFilter,
)
from aug.core.kb.query import UnknownPredicateError
from aug.core.kb.service import KnowledgeBase, get_knowledge_base
from aug.core.tools.context import current_interface, current_thread_id, current_user_id

logger = logging.getLogger(__name__)

_SOURCE_BY_INTERFACE = {"telegram": "telegram", "rest_api": "api"}

_UNAVAILABLE = (
    "Knowledge base is unavailable (not configured or unreachable). NOTHING was saved or "
    "searched — tell the user."
)


@tool(args_schema=KbWriteInput)
async def kb_write(knowledge: str = "", filepath: str = "") -> str:
    """Save information, a file, or both into the knowledge base — the long-term store
    that outlives this conversation. One call can carry several facts. The KB extracts
    the facts, links them to the entities they are about, and makes the text searchable.
    It reads PDFs (scanned too), text files, images and audio itself — pass the filepath
    as is. Other formats return unsupported: kept, but findable only by name.
    Returns the new node id, the facts saved, and status ok / failed / unsupported; if
    `ask` is set, a fact was skipped as ambiguous and that question must be relayed."""
    started = time.monotonic()
    kb, owner = get_knowledge_base(), current_user_id()
    if kb is None or not owner:
        return _log_and_return("kb_write", owner, started, "unavailable", _UNAVAILABLE)
    interface = current_interface()
    result = await kb.write(
        owner,
        knowledge=knowledge,
        filepath=filepath,
        source=_SOURCE_BY_INTERFACE.get(interface, "api"),
        source_ref={"thread_id": current_thread_id(), "interface": interface},
    )
    return _log_and_return(
        "kb_write",
        owner,
        started,
        result.status,
        _json(result),
        facts=len(result.facts),
        file=bool(filepath),
        ask=bool(result.ask),
    )


@tool(args_schema=KbRelateInput)
async def kb_relate(subject_node_id: str, predicate: str, value: Any) -> str:
    """State one fact between things that already exist in the KB, by id — no
    extraction, no guessing. Use when you have the ids (from kb_query / kb_get / a
    kb_write result). Unknown ids are an error; only kb_write creates things. A
    single-valued predicate closes its previous value, which stays as history."""
    started = time.monotonic()
    kb, owner = get_knowledge_base(), current_user_id()
    if kb is None or not owner:
        return _log_and_return("kb_relate", owner, started, "unavailable", _UNAVAILABLE)
    try:
        parsed = _parse_value(value)
        result = await kb.relate(
            owner, subject_node_id=subject_node_id, predicate=predicate, value=parsed
        )
    except (KbError, ValidationError) as exc:
        return _log_and_return("kb_relate", owner, started, "error", f"Error: {exc}")
    return _log_and_return("kb_relate", owner, started, "ok", _json(result))


@tool(args_schema=KbQueryInput)
async def kb_query(query: str = "", filters: list | None = None, limit: int = 10) -> str:
    """Search the knowledge base when you don't have an id: meaning-based and exact-word
    search in one, optionally narrowed by filters (predicate, date, kind, source), all
    AND-ed. Returns hits (node_id, kind, title, snippet, dates), total matches before the
    limit, and undated_excluded — how many a 'dated' filter dropped for having no date,
    which you should mention. Call kb_get on a hit for its full facts, text or file."""
    started = time.monotonic()
    kb, owner = get_knowledge_base(), current_user_id()
    if kb is None or not owner:
        return _log_and_return("kb_query", owner, started, "unavailable", _UNAVAILABLE)
    try:
        request = KbQueryInput.model_validate(
            {"query": query, "filters": [_dump(f) for f in filters or []], "limit": limit}
        )
        result = await kb.query(owner, request)
    except (UnknownPredicateError, ValidationError) as exc:
        return _log_and_return("kb_query", owner, started, "error", f"Error: {exc}")
    return _log_and_return(
        "kb_query",
        owner,
        started,
        "ok",
        _json(result),
        hits=len(result.hits),
        total=result.total,
        query=query[:80],
        filters=len(request.filters),
        top=",".join(h.node_id for h in result.hits[:5]),
    )


@tool(args_schema=KbGetInput)
async def kb_get(node_id: str, as_of: date | None = None, with_text: bool = True) -> str:
    """Fetch everything the KB knows about one node by id: its current facts
    (predicates, facts going out), backlinks (facts pointing at it), mentions, previous
    values, its extracted text, and filepath — the stored original, ready for
    respond_with_file. as_of gives the facts as they were on that date."""
    started = time.monotonic()
    kb, owner = get_knowledge_base(), current_user_id()
    if kb is None or not owner:
        return _log_and_return("kb_get", owner, started, "unavailable", _UNAVAILABLE)
    node = await kb.get(owner, node_id, as_of=as_of, with_text=with_text)
    if node is None:
        return _log_and_return(
            "kb_get",
            owner,
            started,
            "not_found",
            f"No node {node_id!r} in the KB (it may never have existed or was forgotten).",
        )
    return _log_and_return("kb_get", owner, started, "ok", _json(node), node=node_id)


@tool(args_schema=KbForgetInput)
async def kb_forget(node_id: str, reason: str = "") -> str:
    """Forget a node: it disappears from every result, together with its own facts, the
    facts extracted from it, and facts elsewhere pointing at it. Forgetting an entity
    also blanks it out of the text of every item that mentioned it, so forget the
    entity itself (not the items) to drop one detail and keep the rest."""
    started = time.monotonic()
    kb, owner = get_knowledge_base(), current_user_id()
    if kb is None or not owner:
        return _log_and_return("kb_forget", owner, started, "unavailable", _UNAVAILABLE)
    result = await kb.forget(owner, node_id, reason)
    if result is None:
        return _log_and_return(
            "kb_forget",
            owner,
            started,
            "not_found",
            f"No node {node_id!r} in the KB — nothing was forgotten.",
        )
    return _log_and_return("kb_forget", owner, started, "ok", _json(result), node=node_id)


KB_TOOLS: list[BaseTool] = [kb_write, kb_relate, kb_query, kb_get, kb_forget]


async def kb_tools_for_turn(kb: KnowledgeBase | None, owner_id: str) -> list[BaseTool]:
    """KB tools as the model should see them this turn: kb_query's PredicateFilter
    narrowed to the owner's predicates, or left out when there are none yet (an empty
    enum is invalid JSON schema)."""
    if kb is None or not owner_id:
        return KB_TOOLS
    try:
        names = [p.name for p in await kb.predicates(owner_id)]
    except Exception:
        logger.warning("kb_predicates_unavailable owner=%s", owner_id, exc_info=True)
        return KB_TOOLS
    narrowed = kb_query.model_copy(update={"args_schema": _query_schema(names)})
    return [narrowed if t is kb_query else t for t in KB_TOOLS]


def _query_schema(predicates: list[str]) -> type[BaseModel]:
    members: list[type[BaseModel]] = [DateFilter, KindFilter, SourceFilter]
    if predicates:
        narrowed_predicate = create_model(
            "PredicateFilter",
            __base__=PredicateFilter,
            __doc__=PredicateFilter.__doc__,
            predicate=(Literal[tuple(predicates)], ...),  # type: ignore[valid-type]
        )
        members.insert(0, narrowed_predicate)
    filter_type = Union[tuple(members)]  # noqa: UP007 — built at runtime
    return create_model(
        "KbQueryInput",
        __base__=KbQueryInput,
        filters=(
            list[filter_type],  # type: ignore[valid-type]
            Field(default_factory=list, description="Any number of filters, all AND-ed."),
        ),
    )


def _parse_value(value: Any) -> LiteralValue | NodeValue:
    if isinstance(value, LiteralValue | NodeValue):
        return value
    data = _dump(value)
    return (
        NodeValue.model_validate(data)
        if data.get("type") == "node"
        else LiteralValue.model_validate(data)
    )


def _dump(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _json(result: Any) -> str:
    return json.dumps(dataclasses.asdict(result), default=str, ensure_ascii=False)


def _log_and_return(
    tool_name: str, owner: str, started: float, status: str, text: str, **fields: Any
) -> str:
    extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info(
        "kb_tool tool=%s owner=%s status=%s ms=%d %s",
        tool_name,
        owner or "-",
        status,
        (time.monotonic() - started) * 1000,
        extra,
    )
    return text
