"""File storage abstraction.

The ``FileStorage`` base class defines the interface; ``LocalFileStorage``
provides a simple on-disk implementation suitable for development and
single-node deployments.  Swap in an S3 implementation without touching any
router code.
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from aug.api.schemas.files import FileMetadata

logger = logging.getLogger(__name__)


class FileStorage(ABC):
    """Abstract file storage backend."""

    @abstractmethod
    async def save(self, *, file_id: str, filename: str, data: bytes) -> None:
        """Persist *data* under *file_id*."""

    @abstractmethod
    async def get_metadata(self, file_id: str) -> FileMetadata | None:
        """Return metadata for *file_id*, or ``None`` if not found."""

    @abstractmethod
    async def read(self, file_id: str) -> bytes | None:
        """Return raw bytes for *file_id*, or ``None`` if not found."""


class LocalFileStorage(FileStorage):
    """Store files as flat files inside a local directory.

    ``_meta`` is an in-memory index; in production back it with a DB table.
    """

    def __init__(self, base_dir: str | Path = "/tmp/aug_files") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, FileMetadata] = {}

    async def save(self, *, file_id: str, filename: str, data: bytes) -> None:
        dest = self._base / file_id
        dest.write_bytes(data)
        self._meta[file_id] = FileMetadata(
            file_id=file_id,
            filename=filename,
            size_bytes=len(data),
            created_at=datetime.now(tz=UTC),
        )
        logger.debug("stored file_id=%s path=%s", file_id, dest)

    async def get_metadata(self, file_id: str) -> FileMetadata | None:
        return self._meta.get(file_id)

    async def read(self, file_id: str) -> bytes | None:
        path = self._base / file_id
        if not path.exists():
            return None
        return path.read_bytes()
