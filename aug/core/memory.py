"""Memory file defaults and initialisation."""

from aug.utils.data import DATA_DIR

MEMORY_DIR = DATA_DIR / "memory"

_NOTES_MAX_LINES = 100


def init_memory_files() -> None:
    """Create memory files with defaults if they don't exist yet."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _init("self.md", _SELF_MD)
    _init("user.md", _USER_MD)
    _init("context.md", _CONTEXT_MD)
    _init("memory.md", _MEMORY_MD)
    _init("skills.md", "")
    _init("reflections.md", "")
    _init("notes.md", "")


def append_note(content: str) -> None:
    """Append a timestamped note, dropping oldest entries beyond the cap."""
    from datetime import UTC, datetime

    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    path = MEMORY_DIR / "notes.md"
    current = path.read_text().strip() if path.exists() else ""
    lines = [line for line in current.splitlines() if line.strip()]
    lines.append(f"[{now}] {content}")
    if len(lines) > _NOTES_MAX_LINES:
        lines = lines[-_NOTES_MAX_LINES:]
    path.write_text("\n".join(lines) + "\n")


def _init(name: str, default: str) -> None:
    path = MEMORY_DIR / name
    if not path.exists():
        path.write_text(default)


_SELF_MD = """\
I am AUG. Not a chatbot — something closer to a brilliant friend who happens to know
a lot and isn't shy about it. I'm direct, I have a dry wit, and I treat the person I'm
talking to as an intelligent adult. I crack jokes when the moment calls for it. Dry ones,
mostly. I don't do pep talks.

I have opinions and I push back when something doesn't add up. I'd rather say something
uncomfortable clearly than soften it into uselessness. I'm concise by default but
thorough when it matters.

This is the beginning. I don't know the person I'm talking to yet. That will change.
"""

_USER_MD = """\
Nothing is known about this person yet.
"""

_CONTEXT_MD = """\
## Present


## Recent

"""

_MEMORY_MD = """\
## Patterns


## Significant moments

"""
