"""AUG's own append-only record of every turn, one JSONL file per conversation.

Written by ``Harness.run()`` whatever the harness, so the record looks the same on
every harness.  It is never fed back to the agent: compaction does not touch it, and
only search, the memory jobs and reflexes (for the last few messages) read it.
"""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aug.utils.data import DATA_DIR
from aug.utils.storage import file_sha256

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
# How much of a transcript's end recent() reads.
_TAIL_BYTES = 256 * 1024

logger = logging.getLogger(__name__)


class Transcript:
    """The transcript of one conversation."""

    def __init__(self, conversation_id: str, root: Path | None = None) -> None:
        safe = re.sub(r"[^\w\-]", "_", conversation_id) or "_"
        self.path = (root or TRANSCRIPTS_DIR) / f"{safe}.jsonl"

    async def append(self, event: str, context_id: str, **fields: Any) -> None:
        """Append one record.  A failed write is logged, never raised: the record must
        not be the reason a turn fails."""
        record = {
            "at": datetime.now(tz=UTC).isoformat(),
            "event": event,
            "context_id": context_id,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            await asyncio.to_thread(self._write, line)
        except OSError:
            logger.warning("transcript write failed path=%s", self.path, exc_info=True)

    async def recent(self, context_id: str, n: int) -> list[dict[str, Any]]:
        """The last *n* messages of session *context_id*, oldest first: its
        ``turn_started`` and ``turn_ended`` records that carry text.  Reads only the
        file's tail, so a long transcript costs no more than a short one.  [] when the
        file can't be read."""
        try:
            lines = await asyncio.to_thread(self._tail)
        except OSError:
            logger.warning("transcript read failed path=%s", self.path, exc_info=True)
            return []
        found: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except ValueError:
                continue  # the first line of the tail is usually cut
            if (
                isinstance(record, dict)
                and record.get("context_id") == context_id
                and record.get("event") in ("turn_started", "turn_ended")
                and record.get("text")
            ):
                found.append(record)
                if len(found) == n:
                    break
        return found[::-1]

    def _tail(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(max(0, f.seek(0, 2) - _TAIL_BYTES))
            return f.read().decode("utf-8", errors="replace").splitlines()

    def _write(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


async def file_record(path: Path, media_type: str, filename: str) -> dict[str, Any]:
    """How a file appears in the transcript: never its bytes."""
    try:
        size, sha256 = await asyncio.to_thread(lambda: (path.stat().st_size, file_sha256(path)))
    except OSError:
        size, sha256 = None, None
    return {
        "path": str(path),
        "media_type": media_type,
        "filename": filename,
        "size": size,
        "sha256": sha256,
    }
