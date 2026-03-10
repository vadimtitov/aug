"""Files router — upload and retrieve files.

Storage backend is injected via ``aug.utils.storage`` so it can be swapped
(local disk, S3, etc.) without touching this router.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from aug.api.schemas.files import FileMetadata, UploadResponse
from aug.api.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile, request: Request) -> UploadResponse:
    """Upload a file and return its assigned file_id."""
    storage = request.app.state.storage
    file_id = str(uuid.uuid4())
    content = await file.read()

    await storage.save(file_id=file_id, filename=file.filename or "upload", data=content)
    logger.info("uploaded file_id=%s name=%s bytes=%d", file_id, file.filename, len(content))

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "upload",
        size_bytes=len(content),
    )


@router.get("/{file_id}", response_model=FileMetadata)
async def get_file(file_id: str, request: Request) -> FileMetadata:
    """Return metadata for a previously uploaded file."""
    storage = request.app.state.storage
    meta = await storage.get_metadata(file_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return meta
