"""KnowledgeBase — the one object the KB tools and the KB agent talk to.

Built at startup when ``KB_DATABASE_URL`` is set; absent otherwise, and every KB
tool then says so instead of failing obscurely.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aug.config import get_settings
from aug.core.kb.extract import KB_TRANSCRIPTION_MODEL
from aug.core.kb.ingest import Ingestor
from aug.core.kb.models import (
    ForgetResult,
    KbQueryInput,
    LiteralValue,
    Node,
    NodeValue,
    QueryResult,
    RelateResult,
    WriteResult,
)
from aug.core.kb.query import Retriever
from aug.utils.data import DATA_DIR
from aug.utils.kb.db import EMBEDDING_DIMS, create_kb_pool
from aug.utils.kb.embeddings import EmbeddingClient
from aug.utils.kb.store import EntityRow, KbStore, PredicateRow

logger = logging.getLogger(__name__)

KB_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"


@dataclass(frozen=True)
class KbFixedModels:
    """Models the KB uses that are not user-pickable.  The embedding model is fixed
    because every stored vector came from it — another model's vectors are not
    comparable, so switching would need a full re-embed.  Transcription is fixed by
    choice.  Shown read-only next to the pickable ones in the settings UI."""

    embedding: str
    transcription: str


KB_FIXED_MODELS = KbFixedModels(embedding=KB_EMBEDDING_MODEL, transcription=KB_TRANSCRIPTION_MODEL)
KB_OBJECTS_DIR = DATA_DIR / "kb" / "objects"


class KnowledgeBase:
    def __init__(self, store: KbStore, embedder: EmbeddingClient, objects_dir: Path) -> None:
        self._store = store
        self._ingestor = Ingestor(store, embedder, objects_dir)
        self._retriever = Retriever(store, embedder)

    async def write(
        self,
        owner_id: str,
        *,
        knowledge: str,
        filepath: str,
        source: str,
        source_ref: dict[str, Any],
    ) -> WriteResult:
        return await self._ingestor.write(
            owner_id, knowledge=knowledge, filepath=filepath, source=source, source_ref=source_ref
        )

    async def relate(
        self,
        owner_id: str,
        *,
        subject_node_id: str,
        predicate: str,
        value: LiteralValue | NodeValue,
    ) -> RelateResult:
        return await self._ingestor.relate(
            owner_id, subject_node_id=subject_node_id, predicate=predicate, value=value
        )

    async def query(self, owner_id: str, request: KbQueryInput) -> QueryResult:
        return await self._retriever.query(owner_id, request)

    async def get(self, owner_id: str, node_id: str, *, as_of, with_text: bool) -> Node | None:
        return await self._retriever.get(owner_id, node_id, as_of=as_of, with_text=with_text)

    async def forget(self, owner_id: str, node_id: str, reason: str) -> ForgetResult | None:
        return await self._retriever.forget(owner_id, node_id, reason)

    async def predicates(self, owner_id: str) -> list[PredicateRow]:
        async with self._store.session(owner_id) as kb:
            return await kb.list_predicates()

    async def entities(self, owner_id: str, limit: int = 200) -> list[EntityRow]:
        async with self._store.session(owner_id) as kb:
            return await kb.list_entities(limit)

    async def close(self) -> None:
        await self._store.close()


_kb: KnowledgeBase | None = None


async def init_knowledge_base() -> KnowledgeBase | None:
    """Connect the KB if ``KB_DATABASE_URL`` is set.

    An unreachable KB does not block the boot — the rest of AUG keeps working and
    every KB tool reports the KB as unavailable, so nothing is silently lost.
    """
    global _kb
    settings = get_settings()
    if not settings.KB_DATABASE_URL:
        logger.info("KB disabled — KB_DATABASE_URL not set")
        return None
    try:
        pool = await create_kb_pool(settings.KB_DATABASE_URL)
    except Exception:
        logger.exception("KB unavailable — could not connect to KB_DATABASE_URL")
        return None
    embedder = EmbeddingClient(
        model=KB_EMBEDDING_MODEL,
        dims=EMBEDDING_DIMS,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    _kb = KnowledgeBase(KbStore(pool, embed_model=KB_EMBEDDING_MODEL), embedder, KB_OBJECTS_DIR)
    return _kb


def get_knowledge_base() -> KnowledgeBase | None:
    return _kb


def set_knowledge_base(kb: KnowledgeBase | None) -> None:
    global _kb
    _kb = kb
