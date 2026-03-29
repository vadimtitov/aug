"""Unit tests for FileContent — disk persistence, memory cache, and reconstruction."""

import pytest

from aug.api.interfaces.base import FileContent

# ---------------------------------------------------------------------------
# from_bytes
# ---------------------------------------------------------------------------


def test_from_bytes_writes_to_disk(tmp_path):
    path = str(tmp_path / "sub" / "test.jpg")
    fc = FileContent.from_bytes(b"imagedata", path=path, mime_type="image/jpeg")

    assert fc.path == path
    assert fc.filename == "test.jpg"
    assert fc.mime_type == "image/jpeg"
    assert (tmp_path / "sub" / "test.jpg").read_bytes() == b"imagedata"


def test_from_bytes_creates_parent_directories(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "file.txt")
    FileContent.from_bytes(b"hello", path=path, mime_type="text/plain")

    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


def test_from_bytes_sets_transcribe_flag(tmp_path):
    path = str(tmp_path / "voice.ogg")
    fc = FileContent.from_bytes(b"audio", path=path, mime_type="audio/ogg", transcribe=True)
    assert fc.transcribe is True


def test_from_bytes_transcribe_defaults_to_false(tmp_path):
    path = str(tmp_path / "photo.jpg")
    fc = FileContent.from_bytes(b"img", path=path, mime_type="image/jpeg")
    assert fc.transcribe is False


# ---------------------------------------------------------------------------
# filename property
# ---------------------------------------------------------------------------


def test_filename_derived_from_path(tmp_path):
    path = str(tmp_path / "report.docx")
    fc = FileContent(path=path, mime_type="application/octet-stream")
    assert fc.filename == "report.docx"


# ---------------------------------------------------------------------------
# read — cache behaviour
# ---------------------------------------------------------------------------


def test_read_returns_bytes_from_cache(tmp_path, monkeypatch):
    """read() must not hit the disk when bytes are already cached."""
    path = str(tmp_path / "photo.jpg")
    fc = FileContent.from_bytes(b"cached", path=path, mime_type="image/jpeg")

    # Delete the file to prove read() isn't using the disk
    (tmp_path / "photo.jpg").unlink()

    assert fc.read() == b"cached"


def test_read_falls_back_to_disk_when_cache_empty(tmp_path):
    """After plain-constructor deserialisation (no cache), read() reads from disk."""
    path = str(tmp_path / "doc.pdf")
    (tmp_path / "doc.pdf").write_bytes(b"pdfbytes")

    # Simulate checkpoint deserialisation — no data, just fields
    fc = FileContent(path=path, mime_type="application/pdf")

    assert fc.read() == b"pdfbytes"


def test_read_caches_result_after_disk_read(tmp_path):
    """Second read() call must not re-read the disk."""
    path = str(tmp_path / "doc.pdf")
    (tmp_path / "doc.pdf").write_bytes(b"pdfbytes")

    fc = FileContent(path=path, mime_type="application/pdf")
    fc.read()  # populates cache

    # Now delete the file — cache should serve the second call
    (tmp_path / "doc.pdf").unlink()

    assert fc.read() == b"pdfbytes"


def test_read_raises_when_no_cache_and_no_file(tmp_path):
    fc = FileContent(path=str(tmp_path / "missing.txt"), mime_type="text/plain")
    with pytest.raises(FileNotFoundError):
        fc.read()


# ---------------------------------------------------------------------------
# Pydantic serialisation — _cache must not appear in model fields
# ---------------------------------------------------------------------------


def test_cache_not_serialised(tmp_path):
    path = str(tmp_path / "photo.jpg")
    fc = FileContent.from_bytes(b"img", path=path, mime_type="image/jpeg")

    dumped = fc.model_dump()
    assert "_cache" not in dumped
    assert "cache" not in dumped
    assert set(dumped.keys()) == {"path", "mime_type", "transcribe"}


def test_roundtrip_via_model_dump(tmp_path):
    """model_dump() → FileContent(**d) must reconstruct successfully."""
    path = str(tmp_path / "photo.jpg")
    (tmp_path / "photo.jpg").write_bytes(b"img")
    fc = FileContent(path=path, mime_type="image/jpeg")

    dumped = fc.model_dump()
    fc2 = FileContent(**dumped)

    assert fc2.path == fc.path
    assert fc2.filename == fc.filename
    assert fc2.mime_type == fc.mime_type
    assert fc2.read() == b"img"
