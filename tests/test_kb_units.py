"""Unit tests for the knowledge base's pure pieces: embeddings, chunking, extraction
routing, query terms, filter compilation, predicate/slug normalisation."""

import itertools
import json
import math
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from aug.core.kb.extract import (
    CHUNK_TOKENS,
    _vision,
    chunk_text,
    count_tokens,
    extract_text,
    kind_for_mime,
    mime_for,
)
from aug.core.kb.ingest import Ingestor, _ExtractedEntity, _ExtractedFact, _Extraction, _grounded
from aug.core.kb.models import (
    DateFilter,
    KbQueryInput,
    KindFilter,
    LiteralValue,
    PredicateFilter,
)
from aug.core.kb.query import UnknownPredicateError, _compile_filters, _scrubber, query_terms
from aug.core.kb.resolve import normalize_predicate, predicate_key, slugify
from aug.utils.file_settings import AppSettings
from aug.utils.kb.embeddings import EmbeddingClient
from aug.utils.kb.store import EntityEvidence

# ── embeddings ──────────────────────────────────────────────────────────────


def _client(handler) -> EmbeddingClient:
    return EmbeddingClient(
        model="m",
        dims=3,
        base_url="http://proxy/v1/",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )


async def test_embed_truncates_and_renormalises():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert "encoding_format" not in body and "dimensions" not in body
        data = [{"index": i, "embedding": [3.0, 4.0, 0.0, 99.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    vectors = await _client(handler).embed(["a", "b"])
    assert vectors == [[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]]
    assert math.isclose(sum(x * x for x in vectors[0]), 1.0)


async def test_embed_http_error_raises_with_body():
    client = _client(lambda r: httpx.Response(400, text="UnsupportedParamsError"))
    with pytest.raises(RuntimeError, match=r"HTTP 400.*UnsupportedParams"):
        await client.embed(["a"])


async def test_embed_short_response_raises():
    client = _client(lambda r: httpx.Response(200, json={"data": []}))
    with pytest.raises(RuntimeError, match="sent 1 inputs, got 0"):
        await client.embed(["a"])


async def test_embed_too_few_dims_raises():
    client = _client(lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]}))
    with pytest.raises(RuntimeError, match="1 dims, need 3"):
        await client.embed(["a"])


# ── chunking and extraction ─────────────────────────────────────────────────


def test_chunk_short_text_is_one_chunk():
    assert chunk_text("  hello world  ") == ["hello world"]
    assert chunk_text("   ") == []


def test_chunk_long_text_windows_with_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(20))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= CHUNK_TOKENS for c in chunks)
    # Consecutive chunks overlap, so nothing falls between them.
    assert all(any(w in b for w in a.split()[-5:]) for a, b in itertools.pairwise(chunks))


@pytest.mark.parametrize(
    ("mime", "kind"),
    [
        ("application/pdf", "document"),
        ("image/png", "image"),
        ("audio/ogg", "audio"),
        ("video/mp4", "video"),
        ("text/csv", "table"),
        ("text/plain", "document"),
    ],
)
def test_kind_for_mime(mime, kind):
    assert kind_for_mime(mime) == kind


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("a.WEBP", "image/webp"),
        ("v.oga", "audio/ogg"),
        ("c.yml", "application/yaml"),
        ("x.unknownext", "application/octet-stream"),
    ],
)
def test_mime_for_readable_formats_does_not_depend_on_the_host(name, mime):
    assert mime_for(Path(name)) == mime


async def test_yaml_is_read_as_text(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("port: 6543")
    result = await extract_text(path, mime_for(path))
    assert (result.status, result.text) == ("ok", "port: 6543")


async def test_extract_pdf_with_text_layer(tmp_path: Path):
    path = tmp_path / "t.pdf"
    pdf = canvas.Canvas(str(path))
    for i in range(8):
        pdf.drawString(50, 800 - 20 * i, f"Booking reference KX7Q2P line {i} with enough text")
    pdf.save()
    with patch("aug.core.kb.extract._vision", new=AsyncMock()) as vision:
        result = await extract_text(path, "application/pdf")
    assert result.status == "ok" and result.extractor == "pypdf"
    assert "KX7Q2P" in result.text
    vision.assert_not_called()


async def test_extract_scanned_pdf_goes_to_vision(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawImage(ImageReader(Image.new("RGB", (200, 100), "white")), 50, 600)
    pdf.save()
    with patch("aug.core.kb.extract._vision", new=AsyncMock(return_value="TOTAL £37.40")):
        result = await extract_text(path, "application/pdf")
    assert result.status == "ok" and result.extractor == "pypdf+vision"
    assert "37.40" in result.text


async def test_extract_scan_with_empty_vision_is_failed(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawImage(ImageReader(Image.new("RGB", (200, 100), "white")), 50, 600)
    pdf.save()
    with patch("aug.core.kb.extract._vision", new=AsyncMock(return_value="  ")):
        result = await extract_text(path, "application/pdf")
    assert result.status == "failed"


async def test_extract_image_vision_error_is_failed_not_raised(tmp_path: Path):
    path = tmp_path / "x.png"
    path.write_bytes(b"\x89PNG")
    with patch("aug.core.kb.extract._vision", new=AsyncMock(side_effect=TimeoutError("slow"))):
        result = await extract_text(path, "image/png")
    assert result.status == "failed" and "TimeoutError" in result.error


async def test_extract_unknown_mime_is_unsupported(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00\x01")
    result = await extract_text(path, "application/octet-stream")
    assert result.status == "unsupported" and result.text == ""


async def test_extract_plain_text(tmp_path: Path):
    path = tmp_path / "n.md"
    path.write_text("port 6543")
    result = await extract_text(path, "text/markdown")
    assert (result.status, result.text) == ("ok", "port 6543")


# ── configurable models ─────────────────────────────────────────────────────


def _settings(**kb) -> AppSettings:
    return AppSettings.model_validate({"knowledge_base": kb})


async def test_extraction_uses_the_model_set_in_settings_at_call_time():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=_Extraction(title="t", summary="s")
    )
    ingestor = Ingestor(MagicMock(), MagicMock(), Path("/nonexistent"))
    with (
        patch("aug.core.kb.ingest.load_settings", return_value=_settings(extraction_model="x1")),
        patch("aug.core.kb.ingest.build_chat_model", return_value=llm) as build,
    ):
        result, error = await ingestor._extract(
            knowledge="k",
            text="k",
            chunks=["k"],
            filename=None,
            mime=None,
            entities=[],
            predicates=[],
        )
    assert build.call_args.args[0] == "x1"
    assert (result.title, error) == ("t", None)


async def test_extraction_failure_names_the_model_and_keeps_the_write():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=None)
    ingestor = Ingestor(MagicMock(), MagicMock(), Path("/nonexistent"))
    with (
        patch("aug.core.kb.ingest.load_settings", return_value=_settings(extraction_model="x1")),
        patch("aug.core.kb.ingest.build_chat_model", return_value=llm),
    ):
        result, error = await ingestor._extract(
            knowledge="k",
            text="k",
            chunks=["k"],
            filename=None,
            mime=None,
            entities=[],
            predicates=[],
        )
    assert result.facts == [] and error == "x1 returned no structured output"


async def test_vision_uses_the_model_set_in_settings():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=" TOTAL 12 "))
    with (
        patch("aug.core.kb.extract.load_settings", return_value=_settings(vision_model="v1")),
        patch("aug.core.kb.extract.build_chat_model", return_value=llm) as build,
    ):
        assert await _vision(b"img", "image/png") == "TOTAL 12"
    assert build.call_args.args[0] == "v1"


# ── query compilation ───────────────────────────────────────────────────────


def test_query_terms_drop_function_words_keep_tokens():
    assert query_terms("find the booking with KX7Q2P please") == ["booking", "kx7q2p"]
    assert query_terms("когда у Кати день рождения?") == ["когда", "кати", "день", "рождения"]


def test_compile_filters_maps_every_filter():
    request = KbQueryInput(
        filters=[
            PredicateFilter(predicate="phone", value=LiteralValue(value="+44")),
            DateFilter(clock="dated", after=date(2026, 8, 1), include_undated=True),
            KindFilter(kinds=["document"]),
            KindFilter(kinds=["image"]),
        ]
    )
    compiled = _compile_filters(request, {"phone"})
    assert compiled.predicates[0].value_literal == "+44"
    assert compiled.dates[0].include_undated is True
    assert compiled.kinds == ["document", "image"]


def test_compile_filters_rejects_unknown_predicate():
    request = KbQueryInput(filters=[PredicateFilter(predicate="employed_by")])
    with pytest.raises(UnknownPredicateError, match="works_at"):
        _compile_filters(request, {"works_at"})


# ── normalisation ───────────────────────────────────────────────────────────


def test_normalize_predicate():
    assert normalize_predicate("Home Address") == "home_address"
    assert normalize_predicate("works-at!!") == "works_at"
    assert normalize_predicate("  ") == "related_to"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("phone", "phone_number", True),
        ("home_address", "address_home", True),
        ("email_addresses", "email_address", True),
        ("gross_pay_gbp", "net_pay_gbp", False),
        ("home_address", "work_address", False),
        ("address", "previous_address", False),
        ("name", "number", False),  # all filler: compared as is
    ],
)
def test_predicate_key_merges_only_wording_not_meaning(a, b, same):
    assert (predicate_key(a) == predicate_key(b)) is same


def test_slugify_keeps_non_latin_names():
    assert slugify("Anna Logunova") == "anna-logunova"
    assert slugify("Катя") == "катя"
    assert slugify("???") == "unnamed"


# ── grounding ───────────────────────────────────────────────────────────────


def test_grounding_drops_facts_the_text_does_not_support():
    extraction = _Extraction(
        title="t",
        summary="",
        entities=[_ExtractedEntity(ref="e1", type="person", name="Graham Whitlock")],
        facts=[
            _ExtractedFact(
                subject="e1",
                predicate="phone",
                value="+44 7700 900123",
                single_valued=True,
                quote="phone number is +44 7700 900123",
            ),
            _ExtractedFact(
                subject="this",
                predicate="rent",
                value="£1,500",
                single_valued=True,
                quote="rent £1,500",
            ),
            _ExtractedFact(
                subject="me",
                predicate="birthday",
                value="2026-03-12",
                single_valued=True,
                quote="birthday",
            ),
            _ExtractedFact(
                subject="me",
                predicate="landlord",
                value="Graham Whitlock",
                value_ref="e1",
                single_valued=True,
                quote="landlord",
            ),
        ],
    )
    kept, dropped = _grounded(
        extraction,
        "Graham's phone number is +44 7700 900123. Birthday 12 March. Landlord: Graham Whitlock",
    )
    assert dropped == 1
    assert [f.predicate for f in kept.facts] == ["phone", "birthday", "landlord"]


# ── forgetting ──────────────────────────────────────────────────────────────


def test_scrubber_blanks_every_spelling_longest_first():
    scrub = _scrubber(
        EntityEvidence(
            names=["Kastanienallee 45, Berlin, Germany"],
            values=["true"],
            quotes=["lives in Berlin, Germany, at Kastanienallee 45"],
            object_ids=[],
        )
    )
    assert scrub("Katya lives in Berlin, Germany, at Kastanienallee 45.") == "Katya [forgotten]."
    assert scrub("previously at kastanienallee 45, Berlin") == "previously at [forgotten], Berlin"
    assert scrub("it is true") == "it is true"  # generic values are left alone


def test_scrubber_matches_whole_words_and_only_distinctive_values():
    scrub = _scrubber(
        EntityEvidence(
            names=["Ann"], values=["London", "+44 7700 900123"], quotes=[], object_ids=[]
        )
    )
    assert scrub("Ann's Annual report") == "[forgotten]'s Annual report"
    assert scrub("call +44 7700 900123 in London") == "call [forgotten] in London"


def test_scrubber_with_nothing_usable_is_identity():
    scrub = _scrubber(EntityEvidence(names=["X"], values=[], quotes=[], object_ids=[]))
    assert scrub("X marks the spot") == "X marks the spot"
