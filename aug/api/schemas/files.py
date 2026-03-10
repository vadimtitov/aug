"""File upload/retrieval schemas."""

from datetime import datetime

from pydantic import BaseModel


class FileMetadata(BaseModel):
    file_id: str
    filename: str
    mime_type: str | None = None
    size_bytes: int
    created_at: datetime


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
