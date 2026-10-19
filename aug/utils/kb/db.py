"""Connection pool and schema bootstrap for the knowledge base database.

The KB lives in its own Postgres (the ``pgvector/pgvector`` image), reached through
``KB_DATABASE_URL`` — never the shared Postgres behind ``DATABASE_URL``.

Every table carries ``owner_id`` and has row-level security forced on, so a
connection only sees the rows of the owner set with
``set_config('app.owner_id', …, true)`` inside its transaction.  A transaction that
never sets it sees nothing at all.
"""

import logging

import asyncpg
from pgvector.asyncpg import register_vector

from aug.utils.db import strip_driver

logger = logging.getLogger(__name__)

EMBEDDING_DIMS = 1536

# Every KB transaction runs as this role.  The connecting user is usually the
# database superuser (the pgvector image's POSTGRES_USER), and superusers bypass
# row-level security even when it is FORCEd — so sessions drop to a plain role.
KB_SESSION_ROLE = "kb_session"

_KB_TABLES = (
    "kb_object",
    "kb_chunk",
    "kb_entity",
    "kb_predicate",
    "kb_fact",
    "kb_mention",
    "kb_link",
)

_CREATE_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""

_CREATE_TABLES = f"""
CREATE TABLE IF NOT EXISTS kb_object (
    owner_id        TEXT NOT NULL,
    id              TEXT NOT NULL,
    kind            TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    mime            TEXT,
    filename        TEXT,
    bytes_path      TEXT,
    source          TEXT NOT NULL,
    source_ref      JSONB NOT NULL DEFAULT '{{}}',
    received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dated_at        DATE,
    title           TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    knowledge       TEXT NOT NULL DEFAULT '',
    derived_text    TEXT NOT NULL DEFAULT '',
    extractor       TEXT NOT NULL DEFAULT '',
    extract_status  TEXT NOT NULL,
    extract_error   TEXT,
    deleted_at      TIMESTAMPTZ,
    deleted_reason  TEXT,
    PRIMARY KEY (owner_id, id),
    UNIQUE (owner_id, sha256)
);

CREATE TABLE IF NOT EXISTS kb_entity (
    owner_id        TEXT NOT NULL,
    id              TEXT NOT NULL,
    type            TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{{}}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ,
    deleted_reason  TEXT,
    PRIMARY KEY (owner_id, id)
);

-- One row per searchable passage.  node_id is an object (its text, chunked) or an
-- entity (one "profile" row: its name plus its current facts, rebuilt on change).
CREATE TABLE IF NOT EXISTS kb_chunk (
    id              BIGSERIAL PRIMARY KEY,
    owner_id        TEXT NOT NULL,
    node_id         TEXT NOT NULL,
    ord             INT NOT NULL,
    is_profile      BOOLEAN NOT NULL DEFAULT FALSE,
    context         TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL,
    embedding       vector({EMBEDDING_DIMS}) NOT NULL,
    embed_model     TEXT NOT NULL,
    tsv             TSVECTOR GENERATED ALWAYS AS
                        (to_tsvector('simple', context || ' ' || text)) STORED
);

CREATE TABLE IF NOT EXISTS kb_predicate (
    owner_id        TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    value_type      TEXT NOT NULL,
    single_valued   BOOLEAN NOT NULL,
    use_count       INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, name)
);

-- The answerable atoms, and (when value_node_id is set) the graph's edges.
CREATE TABLE IF NOT EXISTS kb_fact (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id         TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    value_literal    TEXT,
    value_node_id    TEXT,
    valid_from       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to         TIMESTAMPTZ,
    recorded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_by    UUID,
    source_object_id TEXT,
    source_quote     TEXT NOT NULL DEFAULT '',
    deleted_at       TIMESTAMPTZ,
    CHECK ((value_literal IS NULL) <> (value_node_id IS NULL))
);

CREATE TABLE IF NOT EXISTS kb_mention (
    owner_id        TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    object_id       TEXT NOT NULL,
    PRIMARY KEY (owner_id, entity_id, object_id)
);

-- Structural, system-generated edges between objects (never LLM-guessed).
CREATE TABLE IF NOT EXISTS kb_link (
    owner_id        TEXT NOT NULL,
    src_id          TEXT NOT NULL,
    dst_id          TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, src_id, dst_id, predicate)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS kb_chunk_embedding_idx
    ON kb_chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_chunk_tsv_idx ON kb_chunk USING gin (tsv);
CREATE INDEX IF NOT EXISTS kb_chunk_node_idx ON kb_chunk (owner_id, node_id);
CREATE INDEX IF NOT EXISTS kb_object_received_idx ON kb_object (owner_id, received_at DESC);
CREATE INDEX IF NOT EXISTS kb_entity_name_trgm_idx
    ON kb_entity USING gin (display_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS kb_fact_subject_idx ON kb_fact (owner_id, subject_id, predicate);
CREATE INDEX IF NOT EXISTS kb_fact_value_node_idx ON kb_fact (owner_id, value_node_id);
"""


async def create_kb_pool(database_url: str) -> asyncpg.Pool:
    """Create the KB pool, with the pgvector codec on every connection, and
    bootstrap the schema.  Idempotent — safe on every startup."""
    dsn = strip_driver(database_url)
    bootstrap = await asyncpg.connect(dsn=dsn)
    try:
        # The vector type must exist before register_vector can find it.
        await bootstrap.execute(_CREATE_EXTENSIONS)
        await bootstrap.execute(_CREATE_TABLES)
        await bootstrap.execute(_CREATE_INDEXES)
        await bootstrap.execute(_rls_sql())
        await bootstrap.execute(_SESSION_ROLE_SQL)
    finally:
        await bootstrap.close()
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=10,
        command_timeout=30,
        init=register_vector,
        server_settings={"application_name": "aug-kb"},
    )
    logger.info("KB Postgres ready — %s", dsn.split("@")[-1])
    return pool


_SESSION_ROLE_SQL = f"""
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{KB_SESSION_ROLE}') THEN
        CREATE ROLE {KB_SESSION_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
END $$;
GRANT {KB_SESSION_ROLE} TO CURRENT_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON {", ".join(_KB_TABLES)} TO {KB_SESSION_ROLE};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {KB_SESSION_ROLE};
"""


def _rls_sql() -> str:
    """Force RLS on every KB table, with one owner policy each.

    FORCE matters: the app connects as the tables' owner, which RLS exempts by
    default.  CREATE POLICY has no IF NOT EXISTS, hence the DO block.
    """
    statements = []
    for table in _KB_TABLES:
        statements.append(f"""
ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {table} FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = '{table}' AND policyname = 'owner_only'
    ) THEN
        CREATE POLICY owner_only ON {table}
            USING (owner_id = current_setting('app.owner_id', true))
            WITH CHECK (owner_id = current_setting('app.owner_id', true));
    END IF;
END $$;""")
    return "\n".join(statements)
