"""Note tool — lightweight mid-conversation memory capture."""

from datetime import UTC, datetime

from langchain_core.tools import tool

from aug.core.memory import MEMORY_DIR


@tool
def note(content: str) -> str:
    """Leave yourself a note about something worth remembering from this conversation.

    Use this when you notice something significant — a fact about the user,
    a disclosure, a shift in mood, a pattern emerging, a clever approach you
    found, something about yourself you want to remember. Keep it brief.
    A separate process will fold it into memory later.

    Not routine information. Things that would matter to someone who knows this
    person well, or that you'd want to remember next time.

    Args:
        content: Brief note to save.
    """
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    path = MEMORY_DIR / "notes.md"
    current = path.read_text() if path.exists() else ""
    path.write_text(current + f"\n[{now}] {content}\n")
    return "Noted."
