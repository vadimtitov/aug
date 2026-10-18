"""Knowledge base against a real pgvector Postgres — the SQL is the thing under test.

The LLM extraction and the embeddings endpoint are replaced (fixed extractions,
deterministic bag-of-words vectors); everything else — schema bootstrap, RLS, the
write transaction, supersession, search, forgetting — runs for real.

Skipped unless KB_TEST_DATABASE_URL points at a pgvector database, e.g. the dev one:
    KB_TEST_DATABASE_URL=postgresql://aug:password@localhost:5433/aug_kb
"""

import hashlib
import math
import os
import re
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

import asyncpg
import pytest

from aug.core.kb.ingest import Ingestor, KbError, _ExtractedEntity, _ExtractedFact, _Extraction
from aug.core.kb.models import (
    DateFilter,
    KbQueryInput,
    KindFilter,
    LiteralValue,
    NodeValue,
    PredicateFilter,
)
from aug.core.kb.query import Retriever
from aug.utils.kb.db import EMBEDDING_DIMS, create_kb_pool
from aug.utils.kb.store import KbStore

DSN = os.environ.get("KB_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not DSN, reason="KB_TEST_DATABASE_URL not set")


class _BagOfWordsEmbedder:
    """Deterministic stand-in: shared words → similar vectors."""

    model = "test-bow"
    dims = EMBEDDING_DIMS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dims
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


@pytest.fixture
async def kb(tmp_path: Path):
    pool = await create_kb_pool(DSN)
    store = KbStore(pool, embed_model="test-bow")
    embedder = _BagOfWordsEmbedder()
    ingestor = Ingestor(store, embedder, tmp_path / "objects")
    retriever = Retriever(store, embedder)
    yield ingestor, retriever, store
    await pool.close()


def _owner() -> str:
    return f"test-{uuid.uuid4().hex[:10]}"


def _extraction(*facts, entities=(), title="t", dated_at=None) -> _Extraction:
    return _Extraction(
        title=title,
        summary="",
        dated_at=dated_at,
        entities=[_ExtractedEntity(**e) for e in entities],
        facts=[_ExtractedFact(**f) for f in facts],
    )


JOHN = {"ref": "e1", "type": "person", "name": "John Bakker"}


async def _write(
    ingestor: Ingestor, owner: str, text: str, extraction: _Extraction, filepath: str = ""
):
    # Facts must be grounded in the text; the fake extraction's quotes always are.
    text = " ".join([text, *(f.quote for f in extraction.facts)])
    with patch.object(Ingestor, "_extract", return_value=(extraction, None)):
        return await ingestor.write(
            owner, knowledge=text, filepath=filepath, source="api", source_ref={}
        )


def _fact(subject, predicate, value, *, single=True, still_true=True, value_ref=None):
    return {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "value_ref": value_ref,
        "single_valued": single,
        "still_true": still_true,
        "quote": value,
    }


async def test_write_then_query_finds_note_and_entity(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    result = await _write(
        ingestor,
        owner,
        "John Bakker lives at 14 Elm Road",
        _extraction(_fact("e1", "home_address", "14 Elm Road"), entities=[JOHN]),
    )
    assert result.status == "ok"
    assert result.facts == ["person:john-bakker home_address 14 Elm Road"]

    found = await retriever.query(owner, KbQueryInput(query="John Bakker address"))
    ids = [h.node_id for h in found.hits]
    assert "person:john-bakker" in ids and result.node_id in ids


async def test_single_valued_fact_supersedes_and_keeps_history(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    await _write(
        ingestor,
        owner,
        "a",
        _extraction(_fact("e1", "home_address", "14 Elm Road"), entities=[JOHN]),
    )
    await _write(
        ingestor,
        owner,
        "b",
        _extraction(_fact("e1", "home_address", "7 Canal St"), entities=[JOHN]),
    )
    node = await retriever.get(owner, "person:john-bakker", as_of=None, with_text=False)
    assert node.predicates["home_address"] == ["7 Canal St"]
    assert any("14 Elm Road" in h for h in node.predicates["(previous values)"])

    then = await retriever.get(owner, "person:john-bakker", as_of=date(2000, 1, 1), with_text=False)
    assert then.predicates == {}  # nothing was known in 2000


async def test_former_value_in_same_write_does_not_become_current(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    await _write(
        ingestor,
        owner,
        "a",
        _extraction(_fact("e1", "home_address", "14 Elm Road"), entities=[JOHN]),
    )
    await _write(
        ingestor,
        owner,
        "moved from 14 Elm Road to 7 Canal St",
        _extraction(
            _fact("e1", "home_address", "7 Canal St"),
            _fact("e1", "home_address", "14 Elm Road", still_true=False),
            entities=[JOHN],
        ),
    )
    node = await retriever.get(owner, "person:john-bakker", as_of=None, with_text=False)
    assert node.predicates["home_address"] == ["7 Canal St"]


async def test_multi_valued_fact_accumulates(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    for city in ("Lisbon", "Porto"):
        await _write(ingestor, owner, city, _extraction(_fact("me", "visited", city, single=False)))
    node = await retriever.get(owner, "person:me", as_of=None, with_text=False)
    assert sorted(node.predicates["visited"]) == ["Lisbon", "Porto"]


async def test_ambiguous_name_skips_fact_and_asks(kb):
    ingestor, _, _ = kb
    owner = _owner()
    for name in ("Anna Logunova", "Anna Schmidt"):
        await _write(
            ingestor,
            owner,
            name,
            _extraction(
                _fact("e1", "phone", f"+44 {len(name)}"),
                entities=[{"ref": "e1", "type": "person", "name": name}],
            ),
        )
    result = await _write(
        ingestor,
        owner,
        "Anna's birthday is 9 Nov",
        _extraction(
            _fact("e1", "birthday", "11-09"),
            entities=[{"ref": "e1", "type": "person", "name": "Anna"}],
        ),
    )
    assert result.status == "ok"
    assert result.facts == []
    assert "Anna Logunova" in result.ask and "Anna Schmidt" in result.ask


async def test_duplicate_note_is_not_stored_twice(kb):
    ingestor, _, _ = kb
    owner = _owner()
    first = await _write(ingestor, owner, "wifi is KettleDrum42", _extraction())
    again = await _write(ingestor, owner, "wifi is KettleDrum42", _extraction())
    assert again.node_id == first.node_id and "Already in the KB" in again.message


async def test_owners_are_isolated_by_rls(kb):
    ingestor, retriever, store = kb
    alice, bob = _owner(), _owner()
    await _write(ingestor, alice, "secret code 4471", _extraction(_fact("me", "code", "4471")))

    found = await retriever.query(bob, KbQueryInput(query="secret code 4471"))
    assert found.hits == [] and found.total == 0
    assert await retriever.get(bob, "person:me", as_of=None, with_text=False) is None
    async with store.session(bob) as session:
        assert await session.list_predicates() == []


async def test_rls_holds_even_for_a_superuser_connection():
    conn = await asyncpg.connect(DSN.replace("+asyncpg", ""))
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE kb_session")
            # No owner set → nothing is visible at all.
            assert await conn.fetchval("SELECT count(*) FROM kb_object") == 0
    finally:
        await conn.close()


async def test_forget_hides_object_its_facts_and_orphan_entities(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    ticket = await _write(
        ingestor,
        owner,
        "parking ticket WB48213977 at Acre Lane",
        _extraction(
            _fact("this", "pcn_number", "WB48213977"),
            _fact("this", "location", "Acre Lane", value_ref="e1"),
            entities=[{"ref": "e1", "type": "place", "name": "Acre Lane"}],
        ),
    )
    keep = await _write(
        ingestor,
        owner,
        "John lives on Elm Road",
        _extraction(_fact("e1", "home_address", "Elm Road"), entities=[JOHN]),
    )

    forgotten = await retriever.forget(owner, ticket.node_id, "asked")
    assert forgotten.facts_hidden == 2

    for query in ("WB48213977", "Acre Lane parking"):
        hits = (await retriever.query(owner, KbQueryInput(query=query))).hits
        assert all(h.node_id not in (ticket.node_id, "place:acre-lane") for h in hits)
    assert await retriever.get(owner, "place:acre-lane", as_of=None, with_text=False) is None
    assert await retriever.get(owner, keep.node_id, as_of=None, with_text=False) is not None
    assert await retriever.forget(owner, ticket.node_id, "again") is None


async def test_filters_apply_inside_search(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    await _write(
        ingestor,
        owner,
        "receipt hardware 37.40",
        _extraction(title="hardware receipt", dated_at="2026-09-12"),
    )
    await _write(ingestor, owner, "receipt dinner 86.50", _extraction(title="dinner receipt"))

    by_dated = await retriever.query(
        owner,
        KbQueryInput(
            query="receipt",
            filters=[DateFilter(clock="dated", after=date(2026, 9, 1), before=date(2026, 10, 1))],
        ),
    )
    assert [h.title for h in by_dated.hits] == ["hardware receipt"]
    assert by_dated.undated_excluded == 1

    by_kind = await retriever.query(owner, KbQueryInput(filters=[KindFilter(kinds=["image"])]))
    assert by_kind.total == 0


async def test_predicate_filter_matches_current_value(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    await _write(
        ingestor, owner, "a", _extraction(_fact("e1", "city", "Amsterdam"), entities=[JOHN])
    )
    found = await retriever.query(
        owner,
        KbQueryInput(
            filters=[
                PredicateFilter(predicate="city", value={"type": "literal", "value": "amster"})
            ]
        ),
    )
    assert [h.node_id for h in found.hits] == ["person:john-bakker"]


async def test_relate_rejects_unknown_nodes_and_supersedes(kb):
    ingestor, _, _ = kb
    owner = _owner()
    await _write(ingestor, owner, "a", _extraction(_fact("e1", "phone", "111"), entities=[JOHN]))
    with pytest.raises(KbError, match="Unknown node"):
        await ingestor.relate(
            owner, subject_node_id="person:nobody", predicate="phone", value=LiteralValue(value="1")
        )
    with pytest.raises(KbError, match="Unknown value node"):
        await ingestor.relate(
            owner,
            subject_node_id="person:john-bakker",
            predicate="knows",
            value=NodeValue(node_id="person:ghost"),
        )
    result = await ingestor.relate(
        owner,
        subject_node_id="person:john-bakker",
        predicate="phone",
        value=LiteralValue(value="222"),
    )
    assert result.predicate_status == "matched" and result.superseded == "111"


async def test_file_is_copied_hashed_and_returned(kb, tmp_path: Path):
    ingestor, retriever, _ = kb
    owner = _owner()
    source = tmp_path / "note.txt"
    source.write_text("booking reference KX7Q2P")
    result = await _write(
        ingestor, owner, "my booking", _extraction(title="booking"), filepath=str(source)
    )
    node = await retriever.get(owner, result.node_id, as_of=None, with_text=True)
    stored = Path(node.filepath)
    assert stored != source and stored.read_text() == "booking reference KX7Q2P"
    assert hashlib.sha256(stored.read_bytes()).hexdigest() in stored.name
    assert "KX7Q2P" in node.text and "my booking" in node.text


async def test_missing_file_fails_honestly(kb):
    ingestor, _, _ = kb
    result = await ingestor.write(
        _owner(), knowledge="", filepath="/nope/x.pdf", source="api", source_ref={}
    )
    assert result.status == "failed" and "NOTHING was saved" in result.message


async def test_similar_predicate_names_never_merge_into_one_value(kb):
    """net_pay must not overwrite gross_pay — the payslip loop."""
    ingestor, retriever, _ = kb
    owner = _owner()
    slip = await _write(
        ingestor,
        owner,
        "payslip",
        _extraction(
            _fact("this", "gross_pay_gbp", "6250.00"),
            _fact("this", "net_pay_gbp", "4210.55"),
        ),
    )
    node = await retriever.get(owner, slip.node_id, as_of=None, with_text=False)
    assert node.predicates["gross_pay_gbp"] == ["6250.00"]
    assert node.predicates["net_pay_gbp"] == ["4210.55"]
    assert "(previous values)" not in node.predicates

    same = await ingestor.relate(
        owner,
        subject_node_id=slip.node_id,
        predicate="pay_gbp_net",
        value=LiteralValue(value="4210.55"),
    )
    assert same.predicate == "net_pay_gbp" and same.superseded is None


async def test_forgetting_an_entity_scrubs_it_from_text_that_named_it(kb):
    ingestor, retriever, _ = kb
    owner = _owner()
    berlin = {"ref": "e2", "type": "place", "name": "Kastanienallee 45, Berlin, Germany"}
    katya = {"ref": "e1", "type": "person", "name": "Katya"}
    note = await _write(
        ingestor,
        owner,
        "my sister Katya lives in Berlin at Kastanienallee 45 and likes cycling",
        _extraction(
            {
                **_fact("e1", "home_address", "Kastanienallee 45", value_ref="e2"),
                "quote": "lives in Berlin at Kastanienallee 45",
            },
            _fact("e1", "hobby", "cycling", single=False),
            entities=[katya, berlin],
        ),
    )
    place_id = "place:kastanienallee-45-berlin-germany"

    result = await retriever.forget(owner, place_id, "asked")
    assert result.inbound_hidden == 1 and note.node_id in result.scrubbed_from

    hits = (await retriever.query(owner, KbQueryInput(query="Kastanienallee 45"))).hits
    assert all("Kastanienallee" not in h.snippet for h in hits)
    assert all(h.node_id != place_id for h in hits)
    text = (await retriever.get(owner, note.node_id, as_of=None, with_text=True)).text
    assert "Kastanienallee" not in text and "[forgotten]" in text
    assert "cycling" in text  # the rest of the item survives
    person = await retriever.get(owner, "person:katya", as_of=None, with_text=False)
    assert person.predicates == {"hobby": ["cycling"]}


async def _stored_chunks(store: KbStore, owner: str) -> list:
    async with store.session(owner) as session:
        return await session._conn.fetch(
            "SELECT node_id, is_profile, context, text, embedding::real[] AS vec FROM kb_chunk"
        )


async def test_every_stored_vector_belongs_to_its_own_text(kb):
    ingestor, _, store = kb
    owner = _owner()
    await _write(
        ingestor,
        owner,
        "John Bakker lives at 14 Elm Road",
        _extraction(_fact("e1", "home_address", "14 Elm Road"), entities=[JOHN]),
    )
    rows = await _stored_chunks(store, owner)
    assert {r["is_profile"] for r in rows} == {True, False}
    for row in rows:
        embedded = row["text"] if row["is_profile"] else f"{row['context']}\n{row['text']}"
        (expected,) = await ingestor._embedder.embed([embedded.strip()])
        assert all(
            math.isclose(a, b, abs_tol=1e-6) for a, b in zip(row["vec"], expected, strict=True)
        )


async def test_write_rebuilds_only_the_profiles_it_changed(kb):
    ingestor, _, store = kb
    owner = _owner()
    acme = {"ref": "e1", "type": "organization", "name": "Acme"}
    beta = {"ref": "e1", "type": "organization", "name": "Beta"}
    await _write(
        ingestor,
        owner,
        "a",
        _extraction(_fact("me", "works_at", "Acme", value_ref="e1"), entities=[acme]),
    )
    embedded: list[str] = []
    real_embed = ingestor._embedder.embed

    async def recording_embed(texts):
        embedded.extend(texts)
        return await real_embed(texts)

    with patch.object(ingestor._embedder, "embed", side_effect=recording_embed):
        await _write(
            ingestor, owner, "b", _extraction(_fact("e1", "phone", "111"), entities=[JOHN])
        )
    assert not any(t.startswith("Me (the user)") for t in embedded)  # "me" untouched

    await _write(
        ingestor,
        owner,
        "c",
        _extraction(_fact("me", "works_at", "Beta", value_ref="e1"), entities=[beta]),
    )
    profiles = {
        r["node_id"]: r["text"] for r in await _stored_chunks(store, owner) if r["is_profile"]
    }
    assert "works at" not in profiles["organization:acme"].lower()  # no longer current
    assert "works at" in profiles["organization:beta"].lower()


async def test_a_failed_write_still_logs_its_timing(kb, caplog):
    ingestor, _, _ = kb
    with (
        patch.object(ingestor._embedder, "embed", side_effect=RuntimeError("embeddings down")),
        caplog.at_level("INFO", logger="aug.core.kb.ingest"),
    ):
        result = await _write(ingestor, _owner(), "note that will fail", _extraction())
    assert result.status == "failed" and "NOTHING was saved" in result.message
    (line,) = [r.getMessage() for r in caplog.records if "kb_write_timing" in r.getMessage()]
    assert "status=failed" in line and "extract=" in line
