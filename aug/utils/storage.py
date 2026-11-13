"""The workspace's file store: every file the user sends and every file AUG sends back.

    files/inbox/YYYY-MM/<id>-<name>    what the user sent (images already normalised)
    files/outbox/YYYY-MM/<id>-<name>   a snapshot of each file sent to the user
    files/index/<id>.json              metadata, so a file can be found by its id

Files are never deleted here and the folder survives /clear, so a path the agent saw
in an earlier turn still opens.  Use the module-level ``file_store`` singleton.
"""

import asyncio
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from aug.utils.data import FILES_DIR
from aug.utils.media import extension_for, normalise_image

logger = logging.getLogger(__name__)

Box = Literal["inbox", "outbox"]

_ID = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class StoredFile:
    id: str
    path: Path
    media_type: str
    filename: str  # the name to show and download it as
    size: int
    sha256: str
    box: Box
    created_at: datetime


class FileStore:
    def __init__(self, root: Path = FILES_DIR) -> None:
        self.root = root

    async def save_inbound(self, data: bytes, filename: str, media_type: str) -> StoredFile:
        """Store a file the user sent.  Images are normalised first (see aug.utils.media),
        and a re-encoded image gets a .jpg name to match."""
        if media_type.startswith("image/"):
            normalised, new_type = await asyncio.to_thread(normalise_image, data, media_type)
            if new_type != media_type:
                filename = f"{Path(filename).stem or 'image'}{extension_for(new_type)}"
            data, media_type = normalised, new_type
        return await asyncio.to_thread(self._write, "inbox", data, filename, media_type)

    async def save_outbound(self, data: bytes, filename: str, media_type: str) -> StoredFile:
        """Store a file AUG made (a generated image) before it is sent."""
        return await asyncio.to_thread(self._write, "outbox", data, filename, media_type)

    async def snapshot_outbound(self, source: Path, filename: str, media_type: str) -> StoredFile:
        """Copy *source* into the outbox, so what was sent stays as it was sent."""
        return await asyncio.to_thread(self._copy, source, filename, media_type)

    def get(self, file_id: str) -> StoredFile | None:
        """The file stored under *file_id*, or None if there is none (or it is gone)."""
        if not _ID.match(file_id):
            return None
        try:
            raw = json.loads((self.root / "index" / f"{file_id}.json").read_text())
        except (FileNotFoundError, ValueError):
            return None
        stored = StoredFile(
            id=raw["id"],
            path=Path(raw["path"]),
            media_type=raw["media_type"],
            filename=raw["filename"],
            size=raw["size"],
            sha256=raw["sha256"],
            box=raw["box"],
            created_at=datetime.fromisoformat(raw["created_at"]),
        )
        return stored if stored.path.is_file() else None

    def _write(self, box: Box, data: bytes, filename: str, media_type: str) -> StoredFile:
        file_id, path = self._new_path(box, filename)
        path.write_bytes(data)
        return self._index(file_id, path, filename, media_type, box)

    def _copy(self, source: Path, filename: str, media_type: str) -> StoredFile:
        file_id, path = self._new_path("outbox", filename)
        shutil.copyfile(source, path)
        return self._index(file_id, path, filename, media_type, "outbox")

    def _new_path(self, box: Box, filename: str) -> tuple[str, Path]:
        file_id = uuid4().hex[:16]
        folder = self.root / box / datetime.now(tz=UTC).strftime("%Y-%m")
        folder.mkdir(parents=True, exist_ok=True)
        return file_id, folder / f"{file_id}-{safe_filename(filename)}"

    def _index(
        self, file_id: str, path: Path, filename: str, media_type: str, box: Box
    ) -> StoredFile:
        stored = StoredFile(
            id=file_id,
            path=path,
            media_type=media_type,
            filename=filename,
            size=path.stat().st_size,
            sha256=file_sha256(path),
            box=box,
            created_at=datetime.now(tz=UTC),
        )
        index = self.root / "index"
        index.mkdir(parents=True, exist_ok=True)
        record = {
            "id": stored.id,
            "path": str(stored.path),
            "media_type": stored.media_type,
            "filename": stored.filename,
            "size": stored.size,
            "sha256": stored.sha256,
            "box": stored.box,
            "created_at": stored.created_at.isoformat(),
        }
        (index / f"{file_id}.json").write_text(json.dumps(record))
        logger.info(
            "file_stored id=%s box=%s type=%s bytes=%d", file_id, box, media_type, stored.size
        )
        return stored


file_store = FileStore()


def safe_filename(filename: str) -> str:
    """*filename* without directories, with only safe characters, at most 200 long."""
    name = Path(filename).name
    safe = re.sub(r"[^\w.\-]", "_", name)
    return safe[:200] if safe.strip("._") else "file"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
