"""System prompt utilities.

Anything that contributes to the system prompt lives here.
"""

from aug.core.tools.memory import get_memory_index
from aug.utils.data import read_data_file


def get_user_context() -> str:
    """Return combined user context for injection into the system prompt.

    Sources (all optional, silently skipped if missing):
      - data/user_profile.md
      - data/custom_instructions.md
      - data/memories.json  (via memory index)
    """
    parts = []

    if profile := _read_md("user_profile.md"):
        parts.append(f"## User Profile\n{profile}")

    if instructions := _read_md("custom_instructions.md"):
        parts.append(f"## Custom Instructions\n{instructions}")

    if index := get_memory_index():
        parts.append(index)

    return "\n\n".join(parts)


def _read_md(name: str) -> str:
    """Read a markdown data file, stripping everything from '## Example' onwards."""
    text = read_data_file(name)
    if "## Example" in text:
        text = text[: text.index("## Example")].strip()
    return text
