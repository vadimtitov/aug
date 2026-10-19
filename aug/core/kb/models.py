"""Knowledge base types: tool inputs (Pydantic, they become the tool schema the model
reads) and results (frozen dataclasses, never a schema).

Docstrings on the input classes ship verbatim in the tool definitions.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

NodeId = str
ObjectKind = Literal["document", "image", "audio", "video", "table", "note", "link"]
SourceKind = Literal["telegram", "email", "api", "web", "upload"]
WriteStatus = Literal["ok", "failed", "unsupported"]


class LiteralValue(BaseModel):
    """A plain value: a number, an amount, a date, a name, an identifier."""

    type: Literal["literal"] = "literal"
    value: str


class NodeValue(BaseModel):
    """Another thing in the KB, by id: an entity or a stored item."""

    type: Literal["node"] = "node"
    node_id: NodeId


# Plain unions (anyOf), not discriminated: models often leave out "type", and each
# variant is already told apart by its required field.
Value = LiteralValue | NodeValue


class PredicateFilter(BaseModel):
    """Match things that carry a given fact. Give the predicate alone to match anything
    that has it at all, or pin the other end with a value. Literal values match as a
    case-insensitive substring."""

    type: Literal["predicate"] = "predicate"
    # Narrowed to a Literal of the owner's real predicates every turn — see
    # aug/core/kb/schemas.py.  Plain str here so stored tool calls always validate.
    predicate: str
    value: Value | None = None


class DateFilter(BaseModel):
    """Restrict by time. Two independent clocks: 'received' is when the KB got the thing
    and always exists; 'dated' is when the thing itself happened, read out of its content,
    and is often unknown. Filtering on 'dated' EXCLUDES anything with no known date — the
    result's undated_excluded says how many; set include_undated=True to keep them, or
    filter on 'received' instead. Pass two of these to constrain both clocks at once.
    `after` is inclusive, `before` is exclusive."""

    type: Literal["date"] = "date"
    clock: Literal["received", "dated"] = Field(
        description=(
            "'received' = when the KB got it (always known); "
            "'dated' = when the thing itself happened (often unknown)"
        )
    )
    after: date | None = None
    before: date | None = None
    include_undated: bool = Field(default=False, description="Only meaningful with clock='dated'.")


class KindFilter(BaseModel):
    """Restrict to certain kinds of stored item. Kinds are OR-ed within one filter.
    Using it leaves out entities."""

    type: Literal["kind"] = "kind"
    kinds: list[ObjectKind]


class SourceFilter(BaseModel):
    """Restrict to which door it came in through."""

    type: Literal["source"] = "source"
    sources: list[SourceKind]


Filter = PredicateFilter | DateFilter | KindFilter | SourceFilter


class KbWriteInput(BaseModel):
    knowledge: str = Field(
        default="",
        description=(
            "The information, stated plainly and self-contained, with full names where known. "
            "One call may carry several facts; each becomes its own row. Given with a file, "
            "it is the file's accompanying note (what it is, where it came from)."
        ),
    )
    filepath: str = Field(
        default="",
        description=(
            "Absolute path to a file to take into the KB (document, image, audio, text), "
            "exactly as it appeared in the message. The file is copied, never moved."
        ),
    )


class KbRelateInput(BaseModel):
    subject_node_id: NodeId = Field(
        description="What the fact is about. Must already exist — an unknown id is an error."
    )
    predicate: str = Field(
        description=(
            "The relationship as snake_case English. The vocabulary is open — invent whatever "
            "fits. A close existing one is reused; the result says which happened."
        )
    )
    value: Value = Field(
        description=(
            "The other end: a literal for a plain value, a node for another thing in the KB "
            "(which must exist too)."
        )
    )


class KbQueryInput(BaseModel):
    query: str = Field(
        default="",
        description=(
            "What it's about, in words. Hybrid search — meaning and exact words both count, "
            "so include distinctive tokens (names, identifiers, numbers). Empty means a pure "
            "filter, newest first."
        ),
    )
    filters: list[Filter] = Field(
        default_factory=list, description="Any number of filters, all AND-ed."
    )
    limit: int = Field(default=10, ge=1, le=50, description="Cap on hits returned.")


class KbGetInput(BaseModel):
    node_id: NodeId = Field(description="The thing to fetch, by id.")
    as_of: date | None = Field(
        default=None,
        description=(
            "Which version of each fact was true on that date. Not about when an item was "
            "received; that is DateFilter."
        ),
    )
    with_text: bool = Field(
        default=True,
        description="Include an object's extracted text. Off when you only need facts or the file.",
    )


class KbForgetInput(BaseModel):
    node_id: NodeId = Field(description="What to forget, by id.")
    reason: str = Field(default="", description="Why — recorded on the tombstone.")


@dataclass(frozen=True)
class WriteResult:
    node_id: NodeId
    kind: str
    facts: list[str]
    entities: list[NodeId]
    status: WriteStatus
    ask: str = ""
    message: str = ""


@dataclass(frozen=True)
class RelateResult:
    fact_id: str
    predicate: str
    predicate_status: Literal["matched", "registered"]
    superseded: str | None


@dataclass(frozen=True)
class Hit:
    node_id: NodeId
    kind: str
    title: str
    snippet: str
    score: float
    received_at: date
    dated_at: date | None


@dataclass(frozen=True)
class QueryResult:
    hits: list[Hit]
    total: int
    undated_excluded: int


@dataclass(frozen=True)
class Node:
    node_id: NodeId
    kind: str
    title: str
    predicates: dict[str, list[str]] = field(default_factory=dict)
    backlinks: dict[str, list[NodeId]] = field(default_factory=dict)
    mentions: list[NodeId] = field(default_factory=list)
    filepath: str | None = None
    text: str | None = None
    received_at: date | None = None
    dated_at: date | None = None


@dataclass(frozen=True)
class ForgetResult:
    node_id: NodeId
    facts_hidden: int
    inbound_hidden: int
    scrubbed_from: list[str] = field(default_factory=list)  # objects whose text named it
