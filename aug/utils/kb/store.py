"""KbStore — typed reads and writes against the knowledge base database.

All SQL for the KB lives here.  Orchestration (``aug/core/kb/``) talks to it through
``KbStore.session(owner)``, which opens one transaction scoped to one owner:

    async with store.session(owner) as kb:
        obj = await kb.get_object("obj:7f2a91c4e0b3")

Node ids are self-describing: ``obj:<12 hex>`` is an object (something the user
sent), anything else (``person:anna-logunova``) is an entity.
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import asyncpg

from aug.utils.kb.db import KB_SESSION_ROLE

OBJECT_PREFIX = "obj:"

# Vector candidates below this cosine similarity are noise, not matches.  Keeps a
# question about something never stored from returning confident-looking hits.
_MIN_VECTOR_SIMILARITY = 0.45
_CANDIDATES_PER_SEARCH = 50
_RRF_K = 60


def is_object_id(node_id: str) -> bool:
    return node_id.startswith(OBJECT_PREFIX)


@dataclass(frozen=True)
class ObjectRow:
    id: str
    kind: str
    sha256: str
    mime: str | None
    filename: str | None
    bytes_path: str | None
    source: str
    received_at: datetime
    dated_at: date | None
    title: str
    summary: str
    knowledge: str
    derived_text: str
    extract_status: str
    deleted_at: datetime | None


@dataclass(frozen=True)
class EntityRow:
    id: str
    type: str
    display_name: str
    aliases: list[str]
    created_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class PredicateRow:
    name: str
    description: str
    value_type: str
    single_valued: bool
    use_count: int


@dataclass(frozen=True)
class FactRow:
    id: str
    subject_id: str
    predicate: str
    value_literal: str | None
    value_node_id: str | None
    valid_from: datetime
    valid_to: datetime | None
    source_object_id: str | None


@dataclass(frozen=True)
class NewObject:
    id: str
    kind: str
    sha256: str
    mime: str | None
    filename: str | None
    bytes_path: str | None
    source: str
    source_ref: dict[str, Any]
    dated_at: date | None
    title: str
    summary: str
    knowledge: str
    derived_text: str
    extractor: str
    extract_status: str
    extract_error: str | None = None


@dataclass(frozen=True)
class NewChunk:
    ord: int
    context: str
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class PredicateCondition:
    """Node must carry a current fact with this predicate (and value, if given)."""

    predicate: str
    value_literal: str | None = None
    value_node_id: str | None = None


@dataclass(frozen=True)
class DateCondition:
    clock: str  # "received" | "dated"
    after: date | None = None
    before: date | None = None
    include_undated: bool = False


@dataclass(frozen=True)
class SearchSpec:
    query_vector: list[float] | None = None
    query_terms: list[str] = field(default_factory=list)
    predicates: list[PredicateCondition] = field(default_factory=list)
    dates: list[DateCondition] = field(default_factory=list)
    kinds: list[str] | None = None
    sources: list[str] | None = None
    limit: int = 10


@dataclass(frozen=True)
class SearchRow:
    node_id: str
    kind: str
    title: str
    snippet: str
    score: float
    received_at: datetime
    dated_at: date | None


@dataclass(frozen=True)
class SearchOutcome:
    rows: list[SearchRow]
    total: int
    undated_excluded: int


@dataclass(frozen=True)
class EntityEvidence:
    """Where an entity's details are written down: its names, the literal values of its
    own facts, the source quotes of every fact involving it, and the live objects that
    mention it or sourced those facts."""

    names: list[str]
    values: list[str]
    quotes: list[str]
    object_ids: list[str]


@dataclass(frozen=True)
class RewrittenChunk:
    id: int
    context: str
    text: str


class KbStore:
    """Owns the KB pool.  Every operation runs inside ``session(owner)``."""

    def __init__(self, pool: asyncpg.Pool, *, embed_model: str) -> None:
        self._pool = pool
        self.embed_model = embed_model

    @asynccontextmanager
    async def session(self, owner_id: str) -> AsyncIterator["KbSession"]:
        if not owner_id:
            raise ValueError("KB session needs an owner id")
        async with self._pool.acquire() as conn, conn.transaction():
            # Row-level security only binds a non-superuser — see KB_SESSION_ROLE.
            await conn.execute(f"SET LOCAL ROLE {KB_SESSION_ROLE}")
            await conn.execute("SELECT set_config('app.owner_id', $1, true)", owner_id)
            # Filtered HNSW scans keep going until they find enough rows that pass
            # the WHERE clause, instead of returning a short list.
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            yield KbSession(conn, owner_id, self.embed_model)

    async def close(self) -> None:
        await self._pool.close()


class KbSession:
    """One owner, one transaction.  Created by ``KbStore.session``."""

    def __init__(self, conn: asyncpg.Connection, owner_id: str, embed_model: str) -> None:
        self._conn = conn
        self.owner_id = owner_id
        self._embed_model = embed_model

    # ── objects ────────────────────────────────────────────────────────────

    async def get_object(
        self, object_id: str, *, include_deleted: bool = False
    ) -> ObjectRow | None:
        row = await self._conn.fetchrow(
            f"SELECT {_OBJECT_COLUMNS} FROM kb_object WHERE id = $1"
            + ("" if include_deleted else " AND deleted_at IS NULL"),
            object_id,
        )
        return _object(row) if row else None

    async def find_object_by_sha(self, sha256: str) -> ObjectRow | None:
        row = await self._conn.fetchrow(
            f"SELECT {_OBJECT_COLUMNS} FROM kb_object WHERE sha256 = $1", sha256
        )
        return _object(row) if row else None

    async def insert_object(self, obj: NewObject) -> None:
        await self._conn.execute(
            """
            INSERT INTO kb_object (owner_id, id, kind, sha256, mime, filename, bytes_path, source,
                source_ref, dated_at, title, summary, knowledge, derived_text, extractor,
                extract_status, extract_error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """,
            self.owner_id,
            obj.id,
            obj.kind,
            obj.sha256,
            obj.mime,
            obj.filename,
            obj.bytes_path,
            obj.source,
            json.dumps(obj.source_ref),
            obj.dated_at,
            obj.title,
            obj.summary,
            obj.knowledge,
            obj.derived_text,
            obj.extractor,
            obj.extract_status,
            obj.extract_error,
        )

    async def undelete_object(self, object_id: str) -> None:
        """Bring back a forgotten object that was sent again."""
        await self._conn.execute(
            "UPDATE kb_object SET deleted_at = NULL, deleted_reason = NULL WHERE id = $1",
            object_id,
        )

    async def insert_chunks(self, node_id: str, chunks: list[NewChunk]) -> None:
        await self._conn.executemany(
            """
            INSERT INTO kb_chunk (owner_id, node_id, ord, context, text, embedding, embed_model)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                (self.owner_id, node_id, c.ord, c.context, c.text, c.embedding, self._embed_model)
                for c in chunks
            ],
        )

    # ── entities ───────────────────────────────────────────────────────────

    async def get_entity(self, entity_id: str) -> EntityRow | None:
        row = await self._conn.fetchrow(
            f"SELECT {_ENTITY_COLUMNS} FROM kb_entity WHERE id = $1 AND deleted_at IS NULL",
            entity_id,
        )
        return _entity(row) if row else None

    async def entity_id_taken(self, entity_id: str) -> bool:
        """True if *entity_id* exists, forgotten or not — ids are never reused."""
        return bool(await self._conn.fetchval("SELECT 1 FROM kb_entity WHERE id = $1", entity_id))

    async def list_entities(self, limit: int = 500) -> list[EntityRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_ENTITY_COLUMNS} FROM kb_entity WHERE deleted_at IS NULL
            ORDER BY created_at LIMIT $1
            """,
            limit,
        )
        return [_entity(r) for r in rows]

    async def entities_named(self, name: str) -> list[EntityRow]:
        """Entities whose display name or an alias equals *name*, ignoring case."""
        rows = await self._conn.fetch(
            f"""
            SELECT {_ENTITY_COLUMNS} FROM kb_entity
            WHERE deleted_at IS NULL
              AND (lower(display_name) = lower($1)
                   OR lower($1) = ANY (SELECT lower(a) FROM unnest(aliases) a))
            """,
            name,
        )
        return [_entity(r) for r in rows]

    async def entities_similar(
        self, name: str, entity_type: str, threshold: float
    ) -> list[EntityRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_ENTITY_COLUMNS} FROM kb_entity
            WHERE deleted_at IS NULL AND type = $2 AND similarity(display_name, $1) >= $3
            ORDER BY similarity(display_name, $1) DESC
            """,
            name,
            entity_type,
            threshold,
        )
        return [_entity(r) for r in rows]

    async def insert_entity(
        self, entity_id: str, entity_type: str, display_name: str, aliases: list[str]
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO kb_entity (owner_id, id, type, display_name, aliases)
            VALUES ($1, $2, $3, $4, $5)
            """,
            self.owner_id,
            entity_id,
            entity_type,
            display_name,
            aliases,
        )

    async def add_aliases(self, entity_id: str, aliases: list[str]) -> None:
        await self._conn.execute(
            """
            UPDATE kb_entity
            SET aliases = ARRAY(SELECT DISTINCT unnest(aliases || $2::text[]))
            WHERE id = $1
            """,
            entity_id,
            aliases,
        )

    async def add_mention(self, entity_id: str, object_id: str) -> None:
        await self._conn.execute(
            """
            INSERT INTO kb_mention (owner_id, entity_id, object_id) VALUES ($1, $2, $3)
            ON CONFLICT DO NOTHING
            """,
            self.owner_id,
            entity_id,
            object_id,
        )

    async def node_exists(self, node_id: str) -> bool:
        if is_object_id(node_id):
            return await self.get_object(node_id) is not None
        return await self.get_entity(node_id) is not None

    async def node_label(self, node_id: str) -> str:
        """Human name for a node: an entity's display name or an object's title."""
        if is_object_id(node_id):
            obj = await self.get_object(node_id, include_deleted=True)
            return obj.title if obj else node_id
        name = await self._conn.fetchval(
            "SELECT display_name FROM kb_entity WHERE id = $1", node_id
        )
        return name or node_id

    # ── predicates ─────────────────────────────────────────────────────────

    async def list_predicates(self) -> list[PredicateRow]:
        rows = await self._conn.fetch(
            """
            SELECT name, description, value_type, single_valued, use_count FROM kb_predicate
            ORDER BY use_count DESC, name
            """
        )
        return [PredicateRow(**dict(r)) for r in rows]

    async def get_predicate(self, name: str) -> PredicateRow | None:
        row = await self._conn.fetchrow(
            """
            SELECT name, description, value_type, single_valued, use_count
            FROM kb_predicate WHERE name = $1
            """,
            name,
        )
        return PredicateRow(**dict(row)) if row else None

    async def insert_predicate(
        self,
        name: str,
        *,
        description: str,
        value_type: str,
        single_valued: bool,
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO kb_predicate (owner_id, name, description, value_type, single_valued)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            """,
            self.owner_id,
            name,
            description,
            value_type,
            single_valued,
        )

    async def bump_predicate(self, name: str) -> None:
        await self._conn.execute(
            "UPDATE kb_predicate SET use_count = use_count + 1 WHERE name = $1", name
        )

    # ── facts ──────────────────────────────────────────────────────────────

    async def current_facts(self, subject_id: str, predicate: str) -> list[FactRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_FACT_COLUMNS} FROM kb_fact
            WHERE subject_id = $1 AND predicate = $2 AND valid_to IS NULL AND deleted_at IS NULL
            """,
            subject_id,
            predicate,
        )
        return [_fact(r) for r in rows]

    async def insert_fact(
        self,
        subject_id: str,
        predicate: str,
        *,
        value_literal: str | None,
        value_node_id: str | None,
        source_object_id: str | None,
        source_quote: str,
    ) -> str:
        fact_id = await self._conn.fetchval(
            """
            INSERT INTO kb_fact (owner_id, subject_id, predicate, value_literal, value_node_id,
                source_object_id, source_quote)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            self.owner_id,
            subject_id,
            predicate,
            value_literal,
            value_node_id,
            source_object_id,
            source_quote,
        )
        return str(fact_id)

    async def close_fact(self, fact_id: str, superseded_by: str | None) -> None:
        """End a fact's validity now — the old value stays answerable via as_of."""
        await self._conn.execute(
            """
            UPDATE kb_fact SET valid_to = NOW(), superseded_by = $2::uuid WHERE id = $1::uuid
            """,
            fact_id,
            superseded_by,
        )

    async def facts_out(self, subject_id: str, as_of: date | None) -> list[FactRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_FACT_COLUMNS} FROM kb_fact
            WHERE subject_id = $1 AND deleted_at IS NULL AND {_valid_at("$2")}
            ORDER BY predicate, valid_from
            """,
            subject_id,
            as_of,
        )
        return [_fact(r) for r in rows]

    async def facts_in(self, node_id: str, as_of: date | None) -> list[FactRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_FACT_COLUMNS} FROM kb_fact
            WHERE value_node_id = $1 AND deleted_at IS NULL AND {_valid_at("$2")}
            ORDER BY predicate, valid_from
            """,
            node_id,
            as_of,
        )
        return [_fact(r) for r in rows]

    async def mentions_of(self, node_id: str) -> list[str]:
        """Objects that name an entity, or entities an object names."""
        if is_object_id(node_id):
            rows = await self._conn.fetch(
                """
                SELECT m.entity_id AS id FROM kb_mention m
                JOIN kb_entity e ON e.id = m.entity_id AND e.deleted_at IS NULL
                WHERE m.object_id = $1 ORDER BY m.entity_id
                """,
                node_id,
            )
        else:
            rows = await self._conn.fetch(
                """
                SELECT m.object_id AS id FROM kb_mention m
                JOIN kb_object o ON o.id = m.object_id AND o.deleted_at IS NULL
                WHERE m.entity_id = $1 ORDER BY o.received_at DESC
                """,
                node_id,
            )
        return [r["id"] for r in rows]

    async def replace_profile(self, entity_id: str, text: str, embedding: list[float]) -> None:
        """Swap an entity's searchable profile row for a freshly built one."""
        await self._conn.execute(
            "DELETE FROM kb_chunk WHERE node_id = $1 AND is_profile", entity_id
        )
        await self._conn.execute(
            """
            INSERT INTO kb_chunk (owner_id, node_id, ord, is_profile, text, embedding, embed_model)
            VALUES ($1, $2, 0, TRUE, $3, $4, $5)
            """,
            self.owner_id,
            entity_id,
            text,
            embedding,
            self._embed_model,
        )

    async def superseded_facts(self, subject_id: str) -> list[FactRow]:
        rows = await self._conn.fetch(
            f"""
            SELECT {_FACT_COLUMNS} FROM kb_fact
            WHERE subject_id = $1 AND valid_to IS NOT NULL AND deleted_at IS NULL
            ORDER BY predicate, valid_to DESC
            """,
            subject_id,
        )
        return [_fact(r) for r in rows]

    async def delete_profile(self, entity_id: str) -> None:
        await self._conn.execute(
            "DELETE FROM kb_chunk WHERE node_id = $1 AND is_profile", entity_id
        )

    # ── forgetting ─────────────────────────────────────────────────────────

    async def forget(self, node_id: str, reason: str) -> tuple[int, int]:
        """Soft-delete a node, its own facts, and facts elsewhere pointing at it.

        Returns (own facts hidden, inbound facts hidden).
        """
        table = "kb_object" if is_object_id(node_id) else "kb_entity"
        await self._conn.execute(
            f"UPDATE {table} SET deleted_at = NOW(), deleted_reason = $2 WHERE id = $1",
            node_id,
            reason,
        )
        own = await self._conn.execute(
            """
            UPDATE kb_fact SET deleted_at = NOW()
            WHERE (subject_id = $1 OR source_object_id = $1) AND deleted_at IS NULL
            """,
            node_id,
        )
        inbound = await self._conn.execute(
            "UPDATE kb_fact SET deleted_at = NOW() WHERE value_node_id = $1 AND deleted_at IS NULL",
            node_id,
        )
        return _rowcount(own), _rowcount(inbound)

    async def forget_orphans_of(self, object_id: str, keep: set[str]) -> list[str]:
        """Forget the entities a forgotten object was the only evidence for.

        An entity survives if any live object still mentions it or any live fact
        still involves it — so only what existed purely because of *object_id*
        (and would otherwise leak its details by name) goes.
        """
        rows = await self._conn.fetch(
            """
            WITH candidates AS (
                SELECT entity_id AS id FROM kb_mention WHERE object_id = $1
                UNION SELECT subject_id FROM kb_fact WHERE source_object_id = $1
                UNION SELECT value_node_id FROM kb_fact
                      WHERE source_object_id = $1 AND value_node_id IS NOT NULL
            )
            UPDATE kb_entity e SET deleted_at = NOW(), deleted_reason = 'only known from ' || $1
            WHERE e.id IN (SELECT id FROM candidates) AND e.deleted_at IS NULL
              AND NOT (e.id = ANY($2::text[]))
              AND NOT EXISTS (
                  SELECT 1 FROM kb_mention m
                  JOIN kb_object o ON o.id = m.object_id AND o.deleted_at IS NULL
                  WHERE m.entity_id = e.id)
              AND NOT EXISTS (
                  SELECT 1 FROM kb_fact f
                  WHERE f.deleted_at IS NULL AND (f.subject_id = e.id OR f.value_node_id = e.id))
            RETURNING e.id
            """,
            object_id,
            sorted(keep),
        )
        return [r["id"] for r in rows]

    async def entity_evidence(self, entity_id: str) -> EntityEvidence:
        """Everything that spells out *entity_id* — call before forgetting it."""
        entity = await self.get_entity(entity_id)
        facts = await self._conn.fetch(
            """
            SELECT subject_id, value_literal, source_quote, source_object_id FROM kb_fact
            WHERE (subject_id = $1 OR value_node_id = $1) AND deleted_at IS NULL
            """,
            entity_id,
        )
        mentioned = await self.mentions_of(entity_id)
        sources = {f["source_object_id"] for f in facts if f["source_object_id"]}
        live_sources = await self._conn.fetch(
            "SELECT id FROM kb_object WHERE id = ANY($1::text[]) AND deleted_at IS NULL",
            sorted(sources),
        )
        return EntityEvidence(
            names=[entity.display_name, *entity.aliases] if entity else [],
            values=[
                f["value_literal"]
                for f in facts
                if f["subject_id"] == entity_id and f["value_literal"]
            ],
            quotes=[f["source_quote"] for f in facts if f["source_quote"]],
            object_ids=sorted({*mentioned, *(r["id"] for r in live_sources)}),
        )

    async def rewrite_objects(
        self, object_ids: list[str], rewrite: Callable[[str], str]
    ) -> list[RewrittenChunk]:
        """Apply *rewrite* to the text fields of objects and their chunks.

        Returns the chunks whose text changed — their embeddings are now stale.
        """
        objects = await self._conn.fetch(
            """
            SELECT id, title, summary, knowledge, derived_text FROM kb_object
            WHERE id = ANY($1::text[])
            """,
            object_ids,
        )
        for o in objects:
            fields = [o["title"], o["summary"], o["knowledge"], o["derived_text"]]
            new = [rewrite(f) for f in fields]
            if new != fields:
                await self._conn.execute(
                    """
                    UPDATE kb_object SET title = $2, summary = $3, knowledge = $4,
                        derived_text = $5
                    WHERE id = $1
                    """,
                    o["id"],
                    *new,
                )
        chunks = await self._conn.fetch(
            """
            SELECT id, context, text FROM kb_chunk
            WHERE node_id = ANY($1::text[]) AND NOT is_profile ORDER BY id
            """,
            object_ids,
        )
        changed = []
        for c in chunks:
            context, text = rewrite(c["context"]), rewrite(c["text"])
            if (context, text) != (c["context"], c["text"]):
                await self._conn.execute(
                    "UPDATE kb_chunk SET context = $2, text = $3 WHERE id = $1",
                    c["id"],
                    context,
                    text,
                )
                changed.append(RewrittenChunk(c["id"], context, text))
        return changed

    async def set_chunk_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        await self._conn.execute(
            "UPDATE kb_chunk SET embedding = $2, embed_model = $3 WHERE id = $1",
            chunk_id,
            embedding,
            self._embed_model,
        )

    async def entities_touched_by(self, node_id: str) -> list[str]:
        """Entities whose facts referenced *node_id* — their profiles need a rebuild."""
        rows = await self._conn.fetch(
            """
            SELECT DISTINCT subject_id AS id FROM kb_fact
            WHERE (value_node_id = $1 OR source_object_id = $1) AND subject_id NOT LIKE 'obj:%'
            UNION
            SELECT DISTINCT value_node_id FROM kb_fact
            WHERE subject_id = $1 AND value_node_id IS NOT NULL AND value_node_id NOT LIKE 'obj:%'
            """,
            node_id,
        )
        return [r["id"] for r in rows if r["id"] and r["id"] != node_id]

    # ── search ─────────────────────────────────────────────────────────────

    async def search(self, spec: SearchSpec) -> SearchOutcome:
        params = _Params()
        where = _node_conditions(spec, params, include_dated=True)
        filtered = f"SELECT * FROM ({_NODES}) n WHERE {' AND '.join(where) or 'TRUE'}"
        undated = await self._count_undated(spec)

        if spec.query_vector is None and not spec.query_terms:
            limit = params.add(spec.limit)
            rows = await self._conn.fetch(
                f"""
                WITH filtered AS ({filtered})
                SELECT f.*, COUNT(*) OVER () AS total,
                       COALESCE(NULLIF(f.summary, ''), left(p.text, 600), '') AS snippet,
                       0.0 AS score
                FROM filtered f
                LEFT JOIN kb_chunk p ON p.node_id = f.node_id AND p.is_profile
                ORDER BY f.received_at DESC LIMIT {limit}
                """,
                *params.values,
            )
            total = rows[0]["total"] if rows else 0
            return SearchOutcome([_search_row(r) for r in rows], total, undated)

        branches = []
        if spec.query_vector is not None:
            vec = params.add(spec.query_vector)
            floor = params.add(_MIN_VECTOR_SIMILARITY)
            branches.append(f"""
                SELECT * FROM (
                    SELECT c.id AS chunk_id, c.node_id,
                           row_number() OVER (ORDER BY c.embedding <=> {vec}) AS rnk
                    FROM kb_chunk c JOIN filtered f ON f.node_id = c.node_id
                    WHERE 1 - (c.embedding <=> {vec}) >= {floor}
                    ORDER BY c.embedding <=> {vec} LIMIT {_CANDIDATES_PER_SEARCH}
                ) v""")
        if spec.query_terms:
            tsq = params.add(" | ".join(spec.query_terms))
            branches.append(f"""
                SELECT * FROM (
                    SELECT c.id AS chunk_id, c.node_id,
                           row_number() OVER (
                               ORDER BY ts_rank_cd(c.tsv, to_tsquery('simple', {tsq})) DESC
                           ) AS rnk
                    FROM kb_chunk c JOIN filtered f ON f.node_id = c.node_id
                    WHERE c.tsv @@ to_tsquery('simple', {tsq})
                    ORDER BY ts_rank_cd(c.tsv, to_tsquery('simple', {tsq})) DESC
                    LIMIT {_CANDIDATES_PER_SEARCH}
                ) t""")
        limit = params.add(spec.limit)
        rows = await self._conn.fetch(
            f"""
            WITH filtered AS ({filtered}),
            ranked AS ({" UNION ALL ".join(branches)}),
            chunk_scores AS (
                SELECT chunk_id, node_id, SUM(1.0 / ({_RRF_K} + rnk)) AS score
                FROM ranked GROUP BY chunk_id, node_id
            ),
            best AS (
                SELECT DISTINCT ON (node_id) node_id, chunk_id, score
                FROM chunk_scores ORDER BY node_id, score DESC
            )
            SELECT f.*, b.score, COUNT(*) OVER () AS total, left(c.text, 600) AS snippet
            FROM best b
            JOIN filtered f ON f.node_id = b.node_id
            JOIN kb_chunk c ON c.id = b.chunk_id
            ORDER BY b.score DESC, f.received_at DESC
            LIMIT {limit}
            """,
            *params.values,
        )
        total = rows[0]["total"] if rows else 0
        return SearchOutcome([_search_row(r) for r in rows], total, undated)

    async def _count_undated(self, spec: SearchSpec) -> int:
        """How many nodes a strict 'dated' filter dropped only for having no date."""
        if not any(d.clock == "dated" and not d.include_undated for d in spec.dates):
            return 0
        params = _Params()
        where = _node_conditions(spec, params, include_dated=False)
        where.append("NOT n.is_entity AND n.dated_at IS NULL")
        return await self._conn.fetchval(
            f"SELECT COUNT(*) FROM ({_NODES}) n WHERE {' AND '.join(where)}", *params.values
        )


# ── SQL fragments and row mappers ──────────────────────────────────────────

_OBJECT_COLUMNS = (
    "id, kind, sha256, mime, filename, bytes_path, source, received_at, dated_at, title, "
    "summary, knowledge, derived_text, extract_status, deleted_at"
)
_ENTITY_COLUMNS = "id, type, display_name, aliases, created_at, deleted_at"
_FACT_COLUMNS = (
    "id::text AS id, subject_id, predicate, value_literal, value_node_id, valid_from, valid_to, "
    "source_object_id"
)

# Every live node, objects and entities alike, in one shape the filters can address.
_NODES = """
    SELECT o.id AS node_id, o.kind, o.title, o.summary, o.received_at, o.dated_at,
           o.source, FALSE AS is_entity
    FROM kb_object o WHERE o.deleted_at IS NULL
    UNION ALL
    SELECT e.id, e.type, e.display_name, '', e.created_at, NULL::date, 'kb', TRUE
    FROM kb_entity e WHERE e.deleted_at IS NULL
"""


class _Params:
    """Collects positional query parameters, handing back their $n placeholders."""

    def __init__(self) -> None:
        self.values: list[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"


def _node_conditions(spec: SearchSpec, params: _Params, *, include_dated: bool) -> list[str]:
    where: list[str] = []
    for p in spec.predicates:
        cond = f"f.predicate = {params.add(p.predicate)}"
        if p.value_literal is not None:
            cond += f" AND f.value_literal ILIKE '%' || {params.add(p.value_literal)} || '%'"
        if p.value_node_id is not None:
            cond += f" AND f.value_node_id = {params.add(p.value_node_id)}"
        where.append(
            "EXISTS (SELECT 1 FROM kb_fact f WHERE f.subject_id = n.node_id "
            f"AND f.deleted_at IS NULL AND f.valid_to IS NULL AND {cond})"
        )
    for d in spec.dates:
        if d.clock == "dated" and not include_dated:
            continue
        # Dates are about things that happened or arrived — entities have neither.
        where.append("NOT n.is_entity")
        column = "n.received_at::date" if d.clock == "received" else "n.dated_at"
        bounds = []
        if d.after:
            bounds.append(f"{column} >= {params.add(d.after)}")
        if d.before:
            bounds.append(f"{column} < {params.add(d.before)}")
        if not bounds:
            continue
        cond = " AND ".join(bounds)
        if d.clock == "dated" and d.include_undated:
            cond = f"(({cond}) OR n.dated_at IS NULL)"
        where.append(cond)
    if spec.kinds is not None:
        where.append(f"n.kind = ANY({params.add(spec.kinds)}::text[])")
    if spec.sources is not None:
        where.append(f"n.source = ANY({params.add(spec.sources)}::text[])")
    return where


def _valid_at(param: str) -> str:
    """Facts valid on date *param*; currently valid ones when it is NULL."""
    return (
        f"(({param}::date IS NULL AND valid_to IS NULL) OR "
        f"({param}::date IS NOT NULL AND valid_from::date <= {param}::date "
        f"AND (valid_to IS NULL OR valid_to::date > {param}::date)))"
    )


def _rowcount(status: str) -> int:
    # asyncpg returns the command tag, e.g. "UPDATE 3".
    return int(status.split()[-1]) if status else 0


def _object(row: asyncpg.Record) -> ObjectRow:
    return ObjectRow(**dict(row))


def _entity(row: asyncpg.Record) -> EntityRow:
    return EntityRow(**{**dict(row), "aliases": list(row["aliases"])})


def _fact(row: asyncpg.Record) -> FactRow:
    return FactRow(**dict(row))


def _search_row(row: asyncpg.Record) -> SearchRow:
    return SearchRow(
        node_id=row["node_id"],
        kind=row["kind"],
        title=row["title"],
        snippet=row["snippet"] or "",
        score=float(row["score"]),
        received_at=row["received_at"],
        dated_at=row["dated_at"],
    )
