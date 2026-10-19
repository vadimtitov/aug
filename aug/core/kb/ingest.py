"""kb_write and kb_relate — everything that puts knowledge in.

kb_write, in order: copy and hash the file, extract its text, chunk it, one
structured LLM call (title, summary, date, per-chunk context lines, facts), then a
single transaction that writes the object, entities, facts (closing superseded
values) and mentions, rebuilds the profiles of every entity whose facts changed, and
stores the chunks — their embedding call runs alongside the transaction, overlapping
the profiles' one.  Each stage's time is logged, failures included.  Always inline;
an overrun is reported as not saved.
"""

import asyncio
import hashlib
import logging
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aug.core.kb.extract import Extracted, chunk_text, extract_text, kind_for_mime, mime_for
from aug.core.kb.models import LiteralValue, NodeValue, RelateResult, WriteResult
from aug.core.kb.resolve import (
    ME_ID,
    ensure_me,
    normalize_predicate,
    refresh_profiles,
    resolve_entity,
    resolve_predicate,
    slugify,
)
from aug.core.llm import build_chat_model
from aug.core.prompts import (
    KB_AMBIGUOUS_ASK,
    KB_EXTRACTION_FILE_BODY,
    KB_EXTRACTION_NOTE_BODY,
    KB_EXTRACTION_SYSTEM,
    KB_EXTRACTION_USER,
)
from aug.utils.file_settings import load_settings
from aug.utils.kb.embeddings import EmbeddingClient
from aug.utils.kb.store import (
    OBJECT_PREFIX,
    EntityRow,
    FactRow,
    KbSession,
    KbStore,
    NewChunk,
    NewObject,
    ObjectRow,
    PredicateRow,
)

logger = logging.getLogger(__name__)


WRITE_TIMEOUT_SECONDS = 150
MAX_FILE_BYTES = 50 * 1024 * 1024
# The extraction call sees at most this much text; chunks past it still get indexed.
_MAX_EXTRACTION_CHARS = 60_000
_MAX_CONTEXT_CHUNKS = 40


class KbError(Exception):
    """A request the KB refuses — the message is what the agent should be told."""


class _ExtractedEntity(BaseModel):
    ref: str = Field(description="Short local handle used in facts, e.g. 'e1'.")
    type: str = Field(
        description="One lowercase noun for what it is: person, organization, place, event, "
        "project, product, concept, …"
    )
    name: str = Field(description="Fullest name known.")
    aliases: list[str] = Field(default_factory=list, description="Other names used in the text.")
    existing_id: str | None = Field(
        default=None, description="Id from KNOWN ENTITIES when this is one of them."
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="Ids of KNOWN ENTITIES this could be, when it is ambiguous.",
    )


class _ExtractedFact(BaseModel):
    subject: str = Field(description='"me", "this", or an entity ref.')
    predicate: str
    value: str
    value_ref: str | None = Field(
        default=None, description="Entity ref when the value is an entity, else null."
    )
    single_valued: bool
    still_true: bool = Field(
        default=True,
        description="false when the text says this value no longer holds (a former one).",
    )
    quote: str


class _Extraction(BaseModel):
    title: str
    summary: str
    dated_at: str | None = None
    entities: list[_ExtractedEntity] = Field(default_factory=list)
    facts: list[_ExtractedFact] = Field(default_factory=list)
    chunk_contexts: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Source:
    kind: str
    ref: dict[str, Any]


@dataclass(frozen=True)
class _Asserted:
    fact_id: str | None  # None when the value already held
    superseded: str | None  # the closed values, for the agent
    closed_nodes: list[str]  # entities that were the closed values — their profiles changed


@dataclass(frozen=True)
class _Graph:
    facts: list[str]
    entity_ids: list[str]
    asks: list[str]
    changed: set[str]  # every node whose profile is now stale


class _StageTimer:
    """Seconds spent in each stage of one write, for the timing log line."""

    def __init__(self) -> None:
        self._started = self._last = time.perf_counter()
        self._laps: list[tuple[str, float]] = []

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self._laps.append((stage, now - self._last))
        self._last = now

    def __str__(self) -> str:
        laps = " ".join(f"{stage}={secs:.2f}s" for stage, secs in self._laps)
        return f"{laps} total={self._last - self._started:.2f}s"


class Ingestor:
    def __init__(self, store: KbStore, embedder: EmbeddingClient, objects_dir: Path) -> None:
        self._store = store
        self._embedder = embedder
        self._objects_dir = objects_dir

    async def write(
        self,
        owner_id: str,
        *,
        knowledge: str,
        filepath: str,
        source: str,
        source_ref: dict[str, Any],
    ) -> WriteResult:
        knowledge = knowledge.strip()
        if not knowledge and not filepath:
            return _failed("Nothing to save: give knowledge, a filepath, or both.")
        try:
            async with asyncio.timeout(WRITE_TIMEOUT_SECONDS):
                return await self._write(
                    owner_id, knowledge, filepath.strip(), _Source(source, source_ref)
                )
        except TimeoutError:
            return _failed(
                f"Took longer than {WRITE_TIMEOUT_SECONDS}s and was stopped. NOTHING was saved."
            )
        except KbError as exc:
            return _failed(str(exc))
        except Exception as exc:
            logger.exception("kb_write_failed owner=%s", owner_id)
            return _failed(f"NOTHING was saved — {type(exc).__name__}: {exc}")

    async def relate(
        self,
        owner_id: str,
        *,
        subject_node_id: str,
        predicate: str,
        value: LiteralValue | NodeValue,
    ) -> RelateResult:
        """One fact between things that exist.  Raises KbError for unknown nodes."""
        name = normalize_predicate(predicate)
        async with self._store.session(owner_id) as kb:
            if subject_node_id == ME_ID:
                await ensure_me(kb)
            if not await kb.node_exists(subject_node_id):
                raise KbError(
                    f"Unknown node {subject_node_id!r}. kb_relate never creates things — "
                    "find the right id with kb_query, or use kb_write."
                )
            value_node = value.node_id if isinstance(value, NodeValue) else None
            if value_node == ME_ID:
                await ensure_me(kb)
            if value_node and not await kb.node_exists(value_node):
                raise KbError(f"Unknown value node {value_node!r}.")
            resolved = await resolve_predicate(
                kb,
                name,
                value_type="node" if value_node else "literal",
                single_valued=value_node is None,
            )
            literal = value.value if isinstance(value, LiteralValue) else None
            asserted = await _assert_fact(
                kb,
                subject_node_id,
                resolved.name,
                single_valued=resolved.single_valued,
                value_literal=literal,
                value_node_id=value_node,
                source_object_id=None,
                quote="",
            )
            await refresh_profiles(
                kb, self._embedder, {subject_node_id, value_node or "", *asserted.closed_nodes}
            )
        return RelateResult(
            fact_id=asserted.fact_id or "",
            predicate=resolved.name,
            predicate_status=resolved.status,
            superseded=asserted.superseded,
        )

    async def _write(
        self, owner_id: str, knowledge: str, filepath: str, source: _Source
    ) -> WriteResult:
        if filepath:
            path = Path(filepath)
            if not path.is_file():
                raise KbError(f"No file at {filepath}. NOTHING was saved.")
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise KbError(f"File is {size // 1_048_576} MB, over the 50 MB limit. Not saved.")
            sha = await asyncio.to_thread(_sha256_file, path)
        else:
            path = None
            sha = hashlib.sha256(knowledge.encode()).hexdigest()
        object_id = OBJECT_PREFIX + sha[:12]

        async with self._store.session(owner_id) as kb:
            existing = await kb.find_object_by_sha(sha)
            if existing and existing.deleted_at is None:
                return _already_stored(existing)
            if existing:
                await kb.undelete_object(existing.id)
                return WriteResult(
                    node_id=existing.id,
                    kind=existing.kind,
                    facts=[],
                    entities=[],
                    status="ok",
                    message="This had been forgotten; it is back in the KB (text only, its "
                    "old facts stay forgotten).",
                )
            snapshots = (await kb.list_entities(), await kb.list_predicates())

        timer = _StageTimer()
        outcome = "failed"
        try:
            result = await self._write_new(
                owner_id, knowledge, path, sha, object_id, source, timer, snapshots
            )
            outcome = result.status
            return result
        finally:
            # Logged whatever happened — the slow writes that time out matter most.
            logger.info(
                "kb_write_timing owner=%s object=%s status=%s %s",
                owner_id,
                object_id,
                outcome,
                timer,
            )

    async def _write_new(
        self,
        owner_id: str,
        knowledge: str,
        path: Path | None,
        sha: str,
        object_id: str,
        source: _Source,
        timer: _StageTimer,
        snapshots: tuple[list[EntityRow], list[PredicateRow]],
    ) -> WriteResult:
        if path is not None:
            mime = mime_for(path)
            stored = await asyncio.to_thread(self._copy_into_store, path, owner_id, sha)
            extracted = await extract_text(stored, mime)
            kind = kind_for_mime(mime)
            filename = path.name
        else:
            mime, stored, filename, kind = None, None, None, "note"
            extracted = Extracted(knowledge, "plain", "ok")
        timer.lap("read")

        chunks = chunk_text(extracted.text)
        extraction, extraction_error = await self._extract(
            knowledge=knowledge,
            text=extracted.text,
            chunks=chunks,
            filename=filename,
            mime=mime,
            entities=snapshots[0],
            predicates=snapshots[1],
        )
        timer.lap("extract")

        extraction, dropped = _grounded(extraction, f"{knowledge}\n{extracted.text}")
        if dropped:
            logger.warning("kb_ungrounded_facts_dropped count=%d", dropped)

        # A file is always findable by its name and what was said about it, even when
        # its text could not be read.
        if filename and not chunks:
            chunks = [f"{filename}\n{knowledge}".strip()]
        contexts = _chunk_contexts(extraction, len(chunks), knowledge if filename else "")
        texts = [f"{ctx}\n{chunk}".strip() for ctx, chunk in zip(contexts, chunks, strict=True)]

        # The chunks don't depend on the graph: embed them alongside the transaction, so
        # their round trip overlaps the profiles' instead of following it.
        chunk_vectors = asyncio.create_task(self._embedder.embed(texts))
        title = extraction.title.strip() or filename or knowledge[:80]
        try:
            async with self._store.session(owner_id) as kb:
                await kb.insert_object(
                    NewObject(
                        id=object_id,
                        kind=kind,
                        sha256=sha,
                        mime=mime,
                        filename=filename,
                        bytes_path=str(stored) if stored else None,
                        source=source.kind,
                        source_ref=source.ref,
                        dated_at=_parse_date(extraction.dated_at),
                        title=title,
                        summary=extraction.summary.strip(),
                        knowledge=knowledge if filename else "",
                        derived_text=extracted.text,
                        extractor=extracted.extractor,
                        extract_status=extracted.status,
                        extract_error=extracted.error,
                    )
                )
                graph = await _write_graph(kb, object_id, extraction)
                timer.lap("graph")
                await refresh_profiles(kb, self._embedder, graph.changed)
                await kb.insert_chunks(
                    object_id,
                    [
                        NewChunk(ord=i, context=ctx, text=chunk, embedding=vec)
                        for i, (ctx, chunk, vec) in enumerate(
                            zip(contexts, chunks, await chunk_vectors, strict=True)
                        )
                    ],
                )
                timer.lap("embed+store")
        finally:
            # If the transaction failed first, stop the embed — or collect its error, so it
            # isn't reported as never retrieved.
            if not chunk_vectors.done():
                chunk_vectors.cancel()
            elif not chunk_vectors.cancelled():
                chunk_vectors.exception()

        return _write_result(
            object_id, kind, graph.facts, graph.entity_ids, graph.asks, extracted, extraction_error
        )

    async def _extract(
        self,
        *,
        knowledge: str,
        text: str,
        chunks: list[str],
        filename: str | None,
        mime: str | None,
        entities: list,
        predicates: list,
    ) -> tuple[_Extraction, str | None]:
        """One structured call.  Failure leaves an empty extraction, never a lost write —
        the text and its chunks are still stored and searchable."""
        if filename:
            shown = chunks[:_MAX_CONTEXT_CHUNKS]
            chunked = "\n\n".join(f"[[chunk {i + 1}]]\n{c}" for i, c in enumerate(shown))
            body = KB_EXTRACTION_FILE_BODY.format(
                filename=filename,
                mime=mime,
                knowledge=knowledge or "(nothing)",
                n_chunks=len(shown),
                chunked_text=chunked[:_MAX_EXTRACTION_CHARS] or "(no text could be read)",
            )
        else:
            body = KB_EXTRACTION_NOTE_BODY.format(knowledge=text[:_MAX_EXTRACTION_CHARS])
        prompt = KB_EXTRACTION_USER.format(
            today=date.today().isoformat(),
            entities="\n".join(
                f"{e.id} | {e.type} | {e.display_name} | {', '.join(e.aliases)}" for e in entities
            )
            or "(none yet)",
            predicates="\n".join(
                f"{p.name} | {p.value_type} | {p.single_valued} | {p.description}"
                for p in predicates
            )
            or "(none yet)",
            body=body,
        )
        model = load_settings().knowledge_base.extraction_model
        llm = build_chat_model(model, temperature=0.0, timeout=120).with_structured_output(
            _Extraction, method="function_calling"
        )
        try:
            result = await llm.ainvoke(
                [("system", KB_EXTRACTION_SYSTEM), ("human", prompt)], config={"callbacks": []}
            )
        except Exception as exc:
            logger.warning("kb_extraction_failed model=%s", model, exc_info=True)
            return _Extraction(title="", summary=""), f"{type(exc).__name__}: {exc}"
        if not isinstance(result, _Extraction):
            return _Extraction(title="", summary=""), f"{model} returned no structured output"
        return result, None

    def _copy_into_store(self, source: Path, owner_id: str, sha: str) -> Path:
        """Copy (never move) into the content-addressed store, then verify the bytes."""
        dest = self._objects_dir / slug_owner(owner_id) / sha[:2] / f"{sha}{source.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".part")
            shutil.copyfile(source, tmp)
            if _sha256_file(tmp) != sha:
                tmp.unlink(missing_ok=True)
                raise KbError("File changed while it was being copied. NOTHING was saved.")
            tmp.rename(dest)
        return dest


def slug_owner(owner_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in owner_id)


async def _write_graph(kb: KbSession, object_id: str, extraction: _Extraction) -> _Graph:
    """Resolve entities and write facts and mentions."""
    refs: dict[str, str | None] = {"me": ME_ID, "this": object_id}
    asks: list[str] = []
    ambiguous: dict[str, tuple[str, tuple[str, ...]]] = {}
    entity_ids: list[str] = []
    for ent in extraction.entities:
        if ent.ref in ("me", "this"):
            continue
        resolution = await resolve_entity(
            kb,
            name=ent.name.strip(),
            entity_type=slugify(ent.type) or "thing",
            aliases=ent.aliases,
            existing_id=ent.existing_id,
            candidates=ent.candidates,
        )
        refs[ent.ref] = resolution.entity_id
        if resolution.entity_id is None:
            ambiguous[ent.ref] = (ent.name, resolution.candidates)
            continue
        if resolution.entity_id not in entity_ids:
            entity_ids.append(resolution.entity_id)
        if resolution.entity_id != ME_ID:
            await kb.add_mention(resolution.entity_id, object_id)

    if any(f.subject == "me" or f.value_ref == "me" for f in extraction.facts):
        await ensure_me(kb)

    facts: list[str] = []
    changed: set[str] = set()
    for fact in extraction.facts:
        blocked = next((r for r in (fact.subject, fact.value_ref) if r in ambiguous), None)
        if blocked:
            name, candidates = ambiguous[blocked]
            options = " or ".join([await kb.node_label(c) for c in candidates])
            ask = KB_AMBIGUOUS_ASK.format(name=name, options=options, quote=fact.quote)
            if ask not in asks:
                asks.append(ask)
            continue
        subject = refs.get(fact.subject)
        value_node = refs.get(fact.value_ref) if fact.value_ref else None
        if subject is None or (fact.value_ref and value_node is None):
            logger.info(
                "kb_fact_dropped unknown ref subject=%s value_ref=%s", fact.subject, fact.value_ref
            )
            continue
        value = fact.value.strip()
        if not value and not value_node:
            continue
        name = normalize_predicate(fact.predicate)
        if not fact.still_true:
            # A former value is history, never the current one: close it if it is
            # still open, otherwise there is nothing to assert.
            for current in await kb.current_facts(subject, name):
                if _same_value(current, value, value_node):
                    await kb.close_fact(current.id, None)
                    changed |= {subject, current.value_node_id or ""}
            continue
        resolved = await resolve_predicate(
            kb,
            name,
            value_type="node" if value_node else "literal",
            single_valued=fact.single_valued,
        )
        asserted = await _assert_fact(
            kb,
            subject,
            resolved.name,
            single_valued=resolved.single_valued,
            value_literal=None if value_node else value,
            value_node_id=value_node,
            source_object_id=object_id,
            quote=fact.quote,
        )
        changed |= {subject, value_node or "", *asserted.closed_nodes}
        shown = await kb.node_label(value_node) if value_node else value
        facts.append(f"{subject} {resolved.name} {shown}")
    # A resolved entity may have gained aliases, so its profile changed too.
    return _Graph(facts, entity_ids, asks, changed | set(entity_ids))


async def _assert_fact(
    kb: KbSession,
    subject_id: str,
    predicate: str,
    *,
    single_valued: bool,
    value_literal: str | None,
    value_node_id: str | None,
    source_object_id: str | None,
    quote: str,
) -> _Asserted:
    """Insert a fact unless it is already current; close what it supersedes."""
    current = await kb.current_facts(subject_id, predicate)
    for fact in current:
        if _same_value(fact, value_literal, value_node_id):
            return _Asserted(fact.id, None, [])
    fact_id = await kb.insert_fact(
        subject_id,
        predicate,
        value_literal=value_literal,
        value_node_id=value_node_id,
        source_object_id=source_object_id,
        source_quote=quote,
    )
    closed = current if single_valued else []
    for fact in closed:
        await kb.close_fact(fact.id, fact_id)
    return _Asserted(
        fact_id,
        ", ".join(f.value_literal or f.value_node_id or "" for f in closed) or None,
        [f.value_node_id for f in closed if f.value_node_id],
    )


def _same_value(fact: FactRow, value_literal: str | None, value_node_id: str | None) -> bool:
    if value_node_id:
        return fact.value_node_id == value_node_id
    return (
        value_literal is not None
        and fact.value_literal is not None
        and fact.value_literal.strip().lower() == value_literal.strip().lower()
    )


def _grounded(extraction: _Extraction, source: str) -> tuple[_Extraction, int]:
    """Drop facts the source text doesn't support — neither their quote nor their
    value appears in it.  The extractor is told never to invent; this makes sure."""
    haystack = _norm(source)
    names = {e.ref: e.name for e in extraction.entities}
    kept = []
    for fact in extraction.facts:
        value = names.get(fact.value_ref, fact.value) if fact.value_ref else fact.value
        evidence = [fact.quote, value, *(_date_forms(value))]
        if any(len(_norm(e)) >= 2 and _norm(e) in haystack for e in evidence):
            kept.append(fact)
    dropped = len(extraction.facts) - len(kept)
    return extraction.model_copy(update={"facts": kept}), dropped


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _date_forms(value: str) -> list[str]:
    """An ISO date the extractor normalised may be written any way in the source."""
    try:
        d = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return []
    return [
        f"{d.day} {d:%B}",
        f"{d:%B} {d.day}",
        f"{d.day} {d:%b}",
        f"{d:%d/%m/%Y}",
        f"{d:%d.%m}",
        f"{d.day:02d}{d.month:02d}",
    ]


def _chunk_contexts(extraction: _Extraction, n_chunks: int, knowledge: str) -> list[str]:
    contexts = list(extraction.chunk_contexts)
    if len(contexts) != n_chunks:
        contexts = [extraction.title] * n_chunks if n_chunks > 1 else [""] * n_chunks
    if knowledge:
        contexts = [f"{knowledge} — {c}" if c else knowledge for c in contexts]
    return contexts


def _write_result(
    object_id: str,
    kind: str,
    facts: list[str],
    entity_ids: list[str],
    asks: list[str],
    extracted: Extracted,
    extraction_error: str | None,
) -> WriteResult:
    notes = []
    status = "ok"
    if extracted.status != "ok":
        status = "unsupported" if extracted.status == "unsupported" else "ok"
        notes.append(
            f"The file was stored but its text could not be read ({extracted.error}); it is "
            "findable only by its name and what was said about it."
        )
    if extraction_error:
        notes.append(f"Stored and searchable, but no facts were extracted ({extraction_error}).")
    return WriteResult(
        node_id=object_id,
        kind=kind,
        facts=facts,
        entities=entity_ids,
        status=status,
        ask=" ".join(asks),
        message=" ".join(notes),
    )


def _already_stored(existing: ObjectRow) -> WriteResult:
    return WriteResult(
        node_id=existing.id,
        kind=existing.kind,
        facts=[],
        entities=[],
        status="ok",
        message=f"Already in the KB since {existing.received_at:%Y-%m-%d} "
        f"({existing.title}); nothing new saved.",
    )


def _failed(message: str) -> WriteResult:
    return WriteResult(node_id="", kind="", facts=[], entities=[], status="failed", message=message)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
