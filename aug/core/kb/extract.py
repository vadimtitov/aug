"""Text out of files, and text into chunks.

Light parsing by design: pypdf for PDFs with a text layer, the vision model for
scanned PDFs and images, Whisper (via the proxy) for audio, plain reads for text.
Anything else is reported ``unsupported`` — never silently stored as empty.
"""

import asyncio
import base64
import io
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from pypdf import PdfReader

from aug.config import get_settings
from aug.core.llm import build_chat_model
from aug.core.prompts import KB_VISION_PROMPT
from aug.utils.file_settings import load_settings

logger = logging.getLogger(__name__)

KB_TRANSCRIPTION_MODEL = "whisper-1"

CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64

# A PDF page with less text than this is treated as a scan.
_MIN_TEXT_CHARS_PER_PAGE = 40
_MAX_SCANNED_PAGES = 10
_TEXT_MIMES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "text/csv",
}
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
# Suffixes whose MIME type must not depend on the host's mimetypes table (older Pythons
# and slim images without /etc/mime.types miss some of these).
_MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".xml": "application/xml",
}

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Extracted:
    text: str
    extractor: str  # "pypdf" | "vision" | "pypdf+vision" | "whisper" | "plain"
    status: str  # "ok" | "failed" | "unsupported"
    error: str | None = None


def mime_for(path: Path) -> str:
    """The MIME type kb_write treats *path* as — fixed for every format it can read."""
    return (
        _MIME_BY_SUFFIX.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def kind_for_mime(mime: str) -> str:
    if mime == "application/pdf":
        return "document"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime in ("text/csv", "application/vnd.ms-excel") or "spreadsheet" in mime:
        return "table"
    return "document"


async def extract_text(path: Path, mime: str) -> Extracted:
    """Pull the text out of the file at *path*.  Never raises for a bad file."""
    try:
        if mime == "application/pdf":
            return await _extract_pdf(path)
        if mime in _IMAGE_MIMES:
            text = await _vision(await asyncio.to_thread(path.read_bytes), mime)
            return Extracted(
                text,
                "vision",
                "ok" if text.strip() else "failed",
                None if text.strip() else "vision model returned no text",
            )
        if mime.startswith("audio/"):
            text = await _transcribe(await asyncio.to_thread(path.read_bytes), mime, path.name)
            return Extracted(
                text,
                "whisper",
                "ok" if text.strip() else "failed",
                None if text.strip() else "transcription was empty",
            )
        if mime.startswith("text/") or mime in _TEXT_MIMES:
            text = await asyncio.to_thread(path.read_text, errors="replace")
            return Extracted(text, "plain", "ok")
    except Exception as exc:
        logger.warning("kb_extract_failed path=%s mime=%s", path, mime, exc_info=True)
        return Extracted("", "", "failed", f"{type(exc).__name__}: {exc}")
    return Extracted("", "", "unsupported", f"no extractor for {mime}")


def chunk_text(text: str) -> list[str]:
    """Split into ~CHUNK_TOKENS windows overlapping by CHUNK_OVERLAP, preferring to
    break on paragraph and sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    tokens = _encoding.encode(text)
    if len(tokens) <= CHUNK_TOKENS:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        piece = _encoding.decode(tokens[start:end])
        if end < len(tokens):
            piece = _trim_to_boundary(piece)
            end = start + len(_encoding.encode(piece))
        chunks.append(piece.strip())
        if end >= len(tokens):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


async def _extract_pdf(path: Path) -> Extracted:
    reader = await asyncio.to_thread(PdfReader, str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    if len(text) >= _MIN_TEXT_CHARS_PER_PAGE * max(1, len(pages)) / 2:
        return Extracted(text, "pypdf", "ok")

    # No usable text layer — a scan.  Each page of a scan is an embedded image.
    images: list[tuple[bytes, str]] = []
    for page in reader.pages[:_MAX_SCANNED_PAGES]:
        for image in page.images:
            images.append((image.data, _mime_for_image_name(image.name)))
    if not images:
        return Extracted(
            text,
            "pypdf",
            "failed" if not text else "ok",
            "PDF has no text layer and no page images" if not text else None,
        )
    described = await asyncio.gather(*(_vision(data, mime) for data, mime in images))
    vision_text = "\n\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(described) if t.strip())
    combined = "\n\n".join(p for p in (text, vision_text) if p)
    if not combined.strip():
        return Extracted("", "pypdf+vision", "failed", "vision model returned no text for scan")
    return Extracted(combined, "pypdf+vision", "ok")


async def _vision(data: bytes, mime: str) -> str:
    llm = build_chat_model(load_settings().knowledge_base.vision_model, temperature=0.0, timeout=90)
    encoded = base64.b64encode(data).decode()
    response = await llm.ainvoke(
        [
            SystemMessage(content=KB_VISION_PROMPT),
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
                ]
            ),
        ],
        config={"callbacks": []},
    )
    return str(response.content).strip()


async def _transcribe(data: bytes, mime: str, filename: str) -> str:
    settings = get_settings()
    async with AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL) as client:
        result = await client.audio.transcriptions.create(
            model=KB_TRANSCRIPTION_MODEL, file=(filename, io.BytesIO(data), mime)
        )
    return result.text


def _mime_for_image_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {".png": "image/png", ".jp2": "image/jp2", ".gif": "image/gif"}.get(suffix, "image/jpeg")


def _trim_to_boundary(piece: str) -> str:
    """Cut *piece* back to its last paragraph or sentence end, if one is near the end."""
    floor = len(piece) // 2
    for sep in ("\n\n", "\n", ". ", "! ", "? "):
        cut = piece.rfind(sep)
        if cut > floor:
            return piece[: cut + len(sep)]
    return piece
