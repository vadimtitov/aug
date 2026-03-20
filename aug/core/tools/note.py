"""Note tool — lightweight mid-conversation memory capture."""

from langchain_core.tools import tool

from aug.core.memory import append_note


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
    append_note(content)
    return "Noted."
