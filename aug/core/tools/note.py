"""Note tool — lightweight mid-conversation memory capture."""

from langchain_core.tools import tool

from aug.core.memory import append_note


@tool
def note(content: str) -> str:
    """Save a note for later memory consolidation.

    For anything that might be useful to remember in a future conversation:
    facts about the user, preferences, operational details, decisions made,
    things learned, patterns noticed, corrections given.

    Keep each note brief and self-contained. A separate process folds notes
    into persistent memory each night; they reach your prompt from the next
    session after that, not in this one.

    NEVER record passwords, tokens, API keys, authentication credentials,
    RTSP URLs with embedded passwords, or anything resembling a secret —
    even if the user mentions it directly.

    Args:
        content: Brief note to save.
    """
    append_note(content)
    return "Noted."
