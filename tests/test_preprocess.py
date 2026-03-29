"""Tests for _preprocess — translating ContentPart lists into LLM message content."""

from unittest.mock import AsyncMock, patch

import pytest

from aug.api.interfaces.base import FileContent, LocationContent, TextContent, _preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_fc(tmp_path, data: bytes = b"imgbytes", mime: str = "image/jpeg") -> FileContent:
    path = str(tmp_path / "photo.jpg")
    return FileContent.from_bytes(data, path=path, mime_type=mime)


def _voice_fc(tmp_path, data: bytes = b"audiobytes") -> FileContent:
    path = str(tmp_path / "voice.ogg")
    return FileContent.from_bytes(data, path=path, mime_type="audio/ogg", transcribe=True)


def _audio_fc(tmp_path, data: bytes = b"songbytes") -> FileContent:
    path = str(tmp_path / "song.mp3")
    return FileContent.from_bytes(data, path=path, mime_type="audio/mpeg")


def _doc_fc(tmp_path, data: bytes = b"docbytes") -> FileContent:
    path = str(tmp_path / "report.docx")
    return FileContent.from_bytes(
        data,
        path=path,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ---------------------------------------------------------------------------
# TextContent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_content_produces_text_block():
    result = await _preprocess([TextContent(text="hello")])
    assert result == "hello"


@pytest.mark.asyncio
async def test_multiple_text_parts_joined_as_string():
    result = await _preprocess([TextContent(text="hello"), TextContent(text="world")])
    assert result == "hello\n\nworld"


# ---------------------------------------------------------------------------
# FileContent — images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_produces_img_marker(tmp_path):
    fc = _image_fc(tmp_path, data=b"imgdata")
    result = await _preprocess([fc])

    assert isinstance(result, str)
    assert "[[img:" in result
    assert fc.path in result
    assert "image/jpeg" in result


@pytest.mark.asyncio
async def test_image_marker_contains_path(tmp_path):
    fc = _image_fc(tmp_path)
    result = await _preprocess([fc])

    assert isinstance(result, str)
    assert fc.path in result


@pytest.mark.asyncio
async def test_image_png_mime_type_in_marker(tmp_path):
    path = str(tmp_path / "img.png")
    fc = FileContent.from_bytes(b"png", path=path, mime_type="image/png")
    result = await _preprocess([fc])

    assert isinstance(result, str)
    assert "image/png" in result


# ---------------------------------------------------------------------------
# FileContent — voice (transcribe=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_transcribed_to_text_block(tmp_path):
    fc = _voice_fc(tmp_path)

    with patch("aug.api.interfaces.base._transcribe", new=AsyncMock(return_value="hello world")):
        result = await _preprocess([fc])

    assert isinstance(result, str)
    assert "hello world" in result


@pytest.mark.asyncio
async def test_voice_injects_path_in_result(tmp_path):
    fc = _voice_fc(tmp_path)

    with patch("aug.api.interfaces.base._transcribe", new=AsyncMock(return_value="transcript")):
        result = await _preprocess([fc])

    assert fc.path in result


@pytest.mark.asyncio
async def test_voice_calls_transcribe_with_correct_bytes(tmp_path):
    fc = _voice_fc(tmp_path, data=b"voicedata")

    with patch(
        "aug.api.interfaces.base._transcribe", new=AsyncMock(return_value="ok")
    ) as mock_transcribe:
        await _preprocess([fc])

    mock_transcribe.assert_called_once_with(b"voicedata", "audio/ogg")


# ---------------------------------------------------------------------------
# FileContent — audio without transcription
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_no_transcription_produces_path_text_only(tmp_path):
    fc = _audio_fc(tmp_path)
    result = await _preprocess([fc])

    assert isinstance(result, str)
    assert fc.path in result
    assert fc.filename in result


@pytest.mark.asyncio
async def test_audio_never_calls_transcribe(tmp_path):
    fc = _audio_fc(tmp_path)

    with patch("aug.api.interfaces.base._transcribe", new=AsyncMock()) as mock_transcribe:
        await _preprocess([fc])

    mock_transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# FileContent — generic document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_produces_path_text_block(tmp_path):
    fc = _doc_fc(tmp_path)
    result = await _preprocess([fc])

    assert isinstance(result, str)
    assert fc.path in result
    assert "report.docx" in result


# ---------------------------------------------------------------------------
# LocationContent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_location_produces_text_block():
    part = LocationContent(latitude=51.5074, longitude=-0.1278)

    with patch(
        "aug.api.interfaces.base._geocode",
        new=AsyncMock(return_value="London, UK"),
    ):
        result = await _preprocess([part])

    assert result == "London, UK"


# ---------------------------------------------------------------------------
# Mixed content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_with_caption_returns_string(tmp_path):
    fc = _image_fc(tmp_path)
    parts = [fc, TextContent(text="What do you see?")]
    result = await _preprocess(parts)

    assert isinstance(result, str)
    assert "[[img:" in result
    assert "What do you see?" in result


@pytest.mark.asyncio
async def test_text_only_parts_return_string(tmp_path):
    """Even when FileContent is present, if it only produces text blocks the
    result must be a plain string (not a list)."""
    fc = _doc_fc(tmp_path)
    parts = [TextContent(text="please process this"), fc]
    result = await _preprocess(parts)

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_image_does_not_call_transcribe(tmp_path):
    fc = _image_fc(tmp_path)
    with patch("aug.api.interfaces.base._transcribe", new=AsyncMock()) as mock_transcribe:
        await _preprocess([fc])
    mock_transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------


def test_safe_filename_strips_path_components():
    from aug.api.interfaces.telegram import _safe_filename

    assert _safe_filename("../../etc/passwd") == "passwd"
    assert _safe_filename("/absolute/path/file.txt") == "file.txt"


def test_safe_filename_replaces_unsafe_chars():
    from aug.api.interfaces.telegram import _safe_filename

    assert _safe_filename("my file (1).docx") == "my_file__1_.docx"


def test_safe_filename_preserves_safe_chars():
    from aug.api.interfaces.telegram import _safe_filename

    assert _safe_filename("report-2026_final.pdf") == "report-2026_final.pdf"


def test_safe_filename_truncates_long_names():
    from aug.api.interfaces.telegram import _safe_filename

    assert len(_safe_filename("a" * 300)) == 200


def test_safe_filename_empty_falls_back():
    from aug.api.interfaces.telegram import _safe_filename

    assert _safe_filename("") == "file"
    assert _safe_filename("...") == "file" or _safe_filename("...") == "..."
