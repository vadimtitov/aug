"""Files router — upload a file for a chat message, and fetch files by id.

Uploads land in the workspace inbox (images normalised), so a ``ChatRequest``
attachment reaches the agent like any file sent from Telegram.  Files the agent sends
are in the outbox and are downloaded from ``GET /files/{file_id}/content``.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from aug.api.schemas.files import FileMetadata, UploadResponse
from aug.api.security import require_api_key
from aug.utils.media import guess_media_type
from aug.utils.storage import StoredFile, file_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(require_api_key)],
)


# Same as what the agent can send back; Telegram's bot upload limit.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile) -> UploadResponse:
    """Upload a file; pass the returned file_id as a ChatRequest attachment."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Files are limited to {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )
    filename = file.filename or "upload"
    media_type = file.content_type
    if not media_type or media_type == "application/octet-stream":
        media_type = guess_media_type(filename)
    stored = await file_store.save_inbound(data, filename, media_type)
    logger.info("uploaded file_id=%s type=%s bytes=%d", stored.id, stored.media_type, stored.size)
    return UploadResponse(
        file_id=stored.id,
        filename=stored.filename,
        mime_type=stored.media_type,
        size_bytes=stored.size,
    )


@router.get("/{file_id}", response_model=FileMetadata)
async def get_file(file_id: str) -> FileMetadata:
    """Metadata for a file, uploaded or sent by the agent."""
    stored = _get(file_id)
    return FileMetadata(
        file_id=stored.id,
        filename=stored.filename,
        mime_type=stored.media_type,
        size_bytes=stored.size,
        created_at=stored.created_at,
    )


@router.get("/{file_id}/content")
async def get_file_content(file_id: str) -> FileResponse:
    """The file's bytes, as an attachment named by its filename."""
    stored = _get(file_id)
    return FileResponse(stored.path, media_type=stored.media_type, filename=stored.filename)


def _get(file_id: str) -> StoredFile:
    stored = file_store.get(file_id)
    if stored is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return stored
