"""AUG's own append-only record of every turn, one JSONL file per conversation.

Written by ``Harness.run()`` whatever the harness, so the record looks the same on
every harness.  It is never fed back to a model: compaction does not touch it, and
only search and the memory jobs read it.
"""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aug.utils.data import DATA_DIR

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

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

    def _write(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
