"""Example tool — returns the current date and time.

Demonstrates the minimal pattern for adding a tool to AUG:
- Decorate a plain function with @tool.
- The docstring becomes the tool description sent to the LLM.
- Type annotations define the input schema.
"""

from datetime import UTC, datetime

from langchain_core.tools import tool


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time.

    Args:
        timezone: ignored for now, always returns UTC.
    """
    return datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
