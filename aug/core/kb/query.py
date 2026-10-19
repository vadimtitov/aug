"""kb_query, kb_get and kb_forget — everything that reads or retracts.

No chat model anywhere here.  kb_query compiles the filters to SQL conditions that
apply inside both the vector and the full-text search, then fuses the two by rank.
"""

import re
from collections.abc import Callable
from datetime import date

from aug.core.kb.models import (
    DateFilter,
    ForgetResult,
    Hit,
    KbQueryInput,
    KindFilter,
    LiteralValue,
    Node,
    NodeValue,
    PredicateFilter,
    QueryResult,
    SourceFilter,
)
from aug.core.kb.resolve import ME_ID, refresh_profiles
from aug.utils.kb.embeddings import EmbeddingClient
from aug.utils.kb.store import (
    DateCondition,
    EntityEvidence,
    KbSession,
    KbStore,
    PredicateCondition,
    SearchSpec,
    is_object_id,
)

FORGOTTEN_MARK = "[forgotten]"

_MAX_TEXT_CHARS = 30_000
_MIN_SCRUB_CHARS = 3
# Literal values too generic to scrub out of surrounding text.
_TRIVIAL_VALUES = frozenset({"true", "false", "yes", "no", "none", "unknown"})

# English function words that carry no meaning for a keyword match (other languages
# lean on the vector side).  Everything else in the query —
# names, codes, numbers — goes into the full-text OR-query.
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from about as into over after before
    is are was were be been being am do does did done have has had i me my mine we our you
    your he she his her it its they them their this that these those what which who whom
    whose when where why how all any some no not can could would should will shall may might
    must there here please tell show find give send get got again now then than so just also
    know remember did ever last much many
    """.split()
)


class UnknownPredicateError(ValueError):
    """A PredicateFilter named a predicate the owner's registry doesn't have."""


class Retriever:
    def __init__(self, store: KbStore, embedder: EmbeddingClient) -> None:
        self._store = store
        self._embedder = embedder

    async def query(self, owner_id: str, request: KbQueryInput) -> QueryResult:
        async with self._store.session(owner_id) as kb:
            known = {p.name for p in await kb.list_predicates()}
        spec = _compile_filters(request, known)
        query = request.query.strip()
        vector = await self._embedder.embed_one(query) if query else None
        terms = query_terms(query)
        async with self._store.session(owner_id) as kb:
            outcome = await kb.search(
                SearchSpec(
                    query_vector=vector,
                    query_terms=terms,
                    predicates=spec.predicates,
                    dates=spec.dates,
                    kinds=spec.kinds,
                    sources=spec.sources,
                    limit=request.limit,
                )
            )
        hits = [
            Hit(
                node_id=r.node_id,
                kind=r.kind,
                title=r.title,
                snippet=r.snippet,
                score=round(r.score, 4),
                received_at=r.received_at.date(),
                dated_at=r.dated_at,
            )
            for r in outcome.rows
        ]
        return QueryResult(
            hits=hits, total=outcome.total, undated_excluded=outcome.undated_excluded
        )

    async def get(
        self, owner_id: str, node_id: str, *, as_of: date | None, with_text: bool
    ) -> Node | None:
        async with self._store.session(owner_id) as kb:
            if is_object_id(node_id):
                obj = await kb.get_object(node_id)
                if obj is None:
                    return None
                kind, title = obj.kind, obj.title
                received, dated = obj.received_at.date(), obj.dated_at
                filepath = obj.bytes_path
                text = None
                if with_text:
                    text = obj.derived_text[:_MAX_TEXT_CHARS]
                    if obj.knowledge:
                        text = f"[What the user said about it: {obj.knowledge}]\n\n{text}"
                    if obj.summary:
                        text = f"[Summary: {obj.summary}]\n\n{text}"
            else:
                entity = await kb.get_entity(node_id)
                if entity is None:
                    return None
                kind, title = entity.type, entity.display_name
                if entity.aliases:
                    title += f" (also: {', '.join(entity.aliases)})"
                received, dated, filepath, text = entity.created_at.date(), None, None, None

            predicates: dict[str, list[str]] = {}
            for fact in await kb.facts_out(node_id, as_of):
                value = fact.value_literal
                if fact.value_node_id:
                    value = f"{fact.value_node_id} ({await kb.node_label(fact.value_node_id)})"
                predicates.setdefault(fact.predicate, []).append(value)
            backlinks: dict[str, list[str]] = {}
            for fact in await kb.facts_in(node_id, as_of):
                backlinks.setdefault(fact.predicate, []).append(fact.subject_id)
            if as_of is None:
                history = await _history(kb, node_id)
                if history:
                    predicates["(previous values)"] = history
            mentions = await kb.mentions_of(node_id)

        return Node(
            node_id=node_id,
            kind=kind,
            title=title,
            predicates=predicates,
            backlinks=backlinks,
            mentions=mentions,
            filepath=filepath,
            text=text,
            received_at=received,
            dated_at=dated,
        )

    async def forget(self, owner_id: str, node_id: str, reason: str) -> ForgetResult | None:
        async with self._store.session(owner_id) as kb:
            if not await kb.node_exists(node_id):
                return None
            touched = set(await kb.entities_touched_by(node_id))
            evidence = None if is_object_id(node_id) else await kb.entity_evidence(node_id)
            own, inbound = await kb.forget(node_id, reason)
            gone = {node_id}
            if is_object_id(node_id):
                gone |= set(await kb.forget_orphans_of(node_id, keep={ME_ID}))
            for entity_id in (n for n in gone if not is_object_id(n)):
                await kb.delete_profile(entity_id)
            scrubbed: list[str] = []
            if evidence:
                # An entity also lives on in the text of whatever mentioned it — scrub
                # it there too, or a full-text or vector hit would bring it straight back.
                scrub = _scrubber(evidence)
                chunks = await kb.rewrite_objects(evidence.object_ids, scrub)
                if chunks:
                    vectors = await self._embedder.embed(
                        [f"{c.context}\n{c.text}".strip() for c in chunks]
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        await kb.set_chunk_embedding(chunk.id, vector)
                scrubbed = evidence.object_ids
            await refresh_profiles(kb, self._embedder, touched - gone)
        return ForgetResult(
            node_id=node_id, facts_hidden=own, inbound_hidden=inbound, scrubbed_from=scrubbed
        )


def query_terms(query: str) -> list[str]:
    """Distinctive words of *query*, safe to OR together in a tsquery."""
    words = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    seen: list[str] = []
    for word in words:
        if (len(word) > 1 or word.isdigit()) and word not in _STOPWORDS and word not in seen:
            seen.append(word)
    return seen[:24]


class _CompiledFilters:
    def __init__(self) -> None:
        self.predicates: list[PredicateCondition] = []
        self.dates: list[DateCondition] = []
        self.kinds: list[str] | None = None
        self.sources: list[str] | None = None


def _compile_filters(request: KbQueryInput, known_predicates: set[str]) -> _CompiledFilters:
    compiled = _CompiledFilters()
    for f in request.filters:
        match f:
            case PredicateFilter(predicate=predicate, value=value):
                if predicate not in known_predicates:
                    raise UnknownPredicateError(
                        f"Unknown predicate {predicate!r}. Known: "
                        f"{', '.join(sorted(known_predicates)) or 'none yet'}."
                    )
                compiled.predicates.append(
                    PredicateCondition(
                        predicate=predicate,
                        value_literal=value.value if isinstance(value, LiteralValue) else None,
                        value_node_id=value.node_id if isinstance(value, NodeValue) else None,
                    )
                )
            case DateFilter(clock=clock, after=after, before=before, include_undated=undated):
                compiled.dates.append(DateCondition(clock, after, before, undated))
            case KindFilter(kinds=kinds):
                compiled.kinds = sorted({*(compiled.kinds or []), *kinds})
            case SourceFilter(sources=sources):
                compiled.sources = list(sources)
    return compiled


async def _history(kb: KbSession, node_id: str) -> list[str]:
    """Superseded values of a node's facts, so "before" questions need no as_of guess."""
    rows = await kb.superseded_facts(node_id)
    out = []
    for fact in rows:
        value = fact.value_literal or f"{fact.value_node_id}"
        until = fact.valid_to.date().isoformat() if fact.valid_to else "?"
        out.append(f"{fact.predicate}: {value} (until {until})")
    return out


def _scrubber(evidence: EntityEvidence) -> Callable[[str], str]:
    """A rewrite that blanks out every spelling of a forgotten entity.

    Longest first, so a whole quote goes before the name inside it.  A name's
    distinctive parts go too ("Kastanienallee 45" out of "Kastanienallee 45, Berlin"):
    texts rarely repeat a full display name verbatim.  Its literal values go only when
    distinctive — a bare "London" or "100" in a text that mentions it is often about
    something else.  Terms match whole words only, so forgetting "Ann" leaves "Annual".
    """
    terms = {*evidence.quotes, *evidence.names}
    terms |= {v for v in evidence.values if _distinctive(v)}
    for name in evidence.names:
        terms |= {part.strip() for part in name.split(",") if _distinctive(part)}
    usable = sorted(
        (t for t in terms if len(t) >= _MIN_SCRUB_CHARS and t.lower() not in _TRIVIAL_VALUES),
        key=len,
        reverse=True,
    )
    if not usable:
        return lambda text: text
    alternatives = "|".join(re.escape(t) for t in usable)
    pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE)
    return lambda text: pattern.sub(FORGOTTEN_MARK, text)


def _distinctive(term: str) -> bool:
    return any(ch.isdigit() for ch in term) or len(term.split()) >= 2
