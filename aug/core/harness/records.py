"""What AUG keeps about a chat's session on an agent CLI: a small JSON file per chat.

The CLI keeps the conversation itself (Claude Code: its session file).  AUG keeps what
that session started with, so it resumes as it started: the prompt, where it replies,
and the last context usage seen, for when the process is closed.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from aug.core.harness.types import Address, ContextUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class SessionRecord:
    """The fields every harness keeps; each adds its own handle on the session."""

    context_id: str
    interface: str
    prompt: str
    conversation_id: str = ""
    chat_id: str = ""
    context: ContextUsage | None = None  # the last seen, for a closed process

    @property
    def reply_to(self) -> Address:
        return Address(
            interface=self.interface, conversation_id=self.conversation_id, chat_id=self.chat_id
        )


class RecordStore[R: SessionRecord]:
    """The records of one harness, of type *kind*, in *folder*.  Writes take turns, so
    an update never overwrites another made while it read."""

    def __init__(self, folder: Path, kind: type[R]) -> None:
        self.folder = folder
        self._kind = kind
        self._writing = asyncio.Lock()

    async def load(self, context_id: str) -> R | None:
        return await asyncio.to_thread(self._read, context_id)

    async def save(self, record: R) -> None:
        async with self._writing:
            await asyncio.to_thread(self._write, record)

    async def update(self, context_id: str, change: Callable[[R], R]) -> R | None:
        """Save *change* applied to the stored record, read and written in one go;
        None (nothing saved) when there is no record.  An unchanged record is not
        written again."""
        async with self._writing:
            record = await asyncio.to_thread(self._read, context_id)
            if record is None:
                return None
            changed = change(record)
            if changed != record:
                await asyncio.to_thread(self._write, changed)
            return changed

    def _path(self, context_id: str) -> Path:
        return self.folder / f"{hashlib.sha256(context_id.encode()).hexdigest()[:32]}.json"

    def _read(self, context_id: str) -> R | None:
        try:
            raw = json.loads(self._path(context_id).read_text())
        except FileNotFoundError:
            return None
        except ValueError:
            logger.warning("unreadable session record context=%s", context_id)
            return None
        known = {f.name for f in fields(self._kind)}
        try:
            data = {k: v for k, v in raw.items() if k in known}
            data["context"] = _context(data.get("context"))
            return self._kind(**data)
        except (AttributeError, TypeError, ValueError):
            logger.warning("unreadable session record context=%s", context_id)
            return None

    def _write(self, record: R) -> None:
        path = self._path(record.context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{uuid.uuid4().hex[:8]}.partial")
        partial.write_text(json.dumps(asdict(record)))
        partial.replace(path)


def _context(raw: Any) -> ContextUsage | None:
    """A stored ContextUsage; None for none, or one that no longer reads."""
    if not isinstance(raw, dict):
        return None
    try:
        breakdown = tuple((str(n), int(t)) for n, t in raw.get("breakdown") or ())
        return ContextUsage(**{**raw, "breakdown": breakdown})
    except (TypeError, ValueError):
        return None
