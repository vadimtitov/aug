"""Unit tests for FileContent — disk persistence and reconstruction."""

import pytest

from aug.api.interfaces.base import FileContent

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_persists_to_disk(tmp_path):
    path = str(tmp_path / "sub" / "test.jpg")
    fc = FileContent(path=path, mime_type="image/jpeg")
    await fc.write(b"imagedata")

    assert (tmp_path / "sub" / "test.jpg").read_bytes() == b"imagedata"


@pytest.mark.asyncio
async def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "file.txt")
    fc = FileContent(path=path, mime_type="text/plain")
    await fc.write(b"hello")

    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


# ---------------------------------------------------------------------------
# filename property
# ---------------------------------------------------------------------------


def test_filename_derived_from_path(tmp_path):
    fc = FileContent(path=str(tmp_path / "report.docx"), mime_type="application/octet-stream")
    assert fc.filename == "report.docx"


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_returns_bytes_from_disk(tmp_path):
    path = str(tmp_path / "doc.pdf")
    (tmp_path / "doc.pdf").write_bytes(b"pdfbytes")

    fc = FileContent(path=path, mime_type="application/pdf")
    assert await fc.read() == b"pdfbytes"


@pytest.mark.asyncio
async def test_read_raises_when_file_missing(tmp_path):
    fc = FileContent(path=str(tmp_path / "missing.txt"), mime_type="text/plain")
    with pytest.raises(FileNotFoundError):
        await fc.read()


@pytest.mark.asyncio
async def test_write_then_read_roundtrip(tmp_path):
    fc = FileContent(path=str(tmp_path / "file.bin"), mime_type="application/octet-stream")
    await fc.write(b"roundtrip")
    assert await fc.read() == b"roundtrip"


# ---------------------------------------------------------------------------
# transcribe flag
# ---------------------------------------------------------------------------


def test_transcribe_defaults_to_false(tmp_path):
    fc = FileContent(path=str(tmp_path / "photo.jpg"), mime_type="image/jpeg")
    assert fc.transcribe is False


def test_transcribe_flag_set(tmp_path):
    fc = FileContent(path=str(tmp_path / "voice.ogg"), mime_type="audio/ogg", transcribe=True)
    assert fc.transcribe is True


# ---------------------------------------------------------------------------
# Pydantic serialisation
# ---------------------------------------------------------------------------


def test_model_dump_contains_expected_keys(tmp_path):
    fc = FileContent(path=str(tmp_path / "photo.jpg"), mime_type="image/jpeg")
    assert set(fc.model_dump().keys()) == {"path", "mime_type", "transcribe"}


def test_roundtrip_via_model_dump(tmp_path):
    fc = FileContent(path=str(tmp_path / "photo.jpg"), mime_type="image/jpeg")
    fc2 = FileContent(**fc.model_dump())
    assert fc2.path == fc.path
    assert fc2.filename == fc.filename
    assert fc2.mime_type == fc.mime_type
