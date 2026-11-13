"""The one way a file reaches the user: send a file that is already on disk."""

import logging
from pathlib import Path

from langchain_core.tools import tool

from aug.core.tools.blocked_paths import path_block_reason
from aug.core.tools.context import current_tool_context
from aug.utils.media import guess_media_type
from aug.utils.storage import StoredFile, file_store

logger = logging.getLogger(__name__)

# Telegram's limit for files a bot uploads.
MAX_FILE_BYTES = 50 * 1024 * 1024


@tool
async def respond_with_file(path: str, caption: str = "") -> str:
    """Send a file on disk to the user: a photo, a PDF, a CSV, a generated document,
    an audio clip. Images, audio and video are shown in the chat as media; anything
    else arrives as a downloadable file.

    To send text you have written, save it to a file first (e.g. with run_bash), then
    pass its path here.

    Args:
        path: Absolute path to the file, e.g. one written by run_bash or a file the
            user sent earlier.
        caption: Optional short text shown with the file.
    """
    file = Path(path).resolve()  # what is checked is what gets copied, links resolved
    if reason := path_block_reason(file):
        return f"File NOT sent. {reason}"
    if not file.is_file():
        return f"File NOT sent: there is no file at {path}."
    size = file.stat().st_size
    if size == 0:
        return f"File NOT sent: {path} is empty."
    if size > MAX_FILE_BYTES:
        return (
            f"File NOT sent: {path} is {size / 1024 / 1024:.1f} MB, over the "
            f"{MAX_FILE_BYTES // 1024 // 1024} MB limit. Compress or split it first."
        )
    if current_tool_context() is None:
        return "File NOT sent: there is no conversation to send it to."

    stored = await send_path(file, caption=caption)
    logger.info("respond_with_file sent %s as %s (%d bytes)", path, stored.id, size)
    return f"Sent {file.name} to the user ({stored.media_type}, {size} bytes)."


async def send_path(file: Path, filename: str = "", caption: str = "") -> StoredFile:
    """Snapshot *file* into the outbox and send it to the user from the running tool.

    For AUG's own tools, which have already decided the file may go out.  Outside a
    turn the snapshot is still made, and nothing is sent.
    """
    filename = filename or file.name
    stored = await file_store.snapshot_outbound(file, filename, guess_media_type(filename))
    ctx = current_tool_context()
    if ctx is not None:
        await ctx.send_file(stored, caption)
    return stored
