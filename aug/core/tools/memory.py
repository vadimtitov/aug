"""Memory tools — remember, recall, update, forget.

Memories are stored in data/memories.json as a list of objects with id,
topic, and content. The index (id + topic) is injected into every system
prompt so the agent can decide which memories to recall or update.
"""

import json
import uuid

from langchain_core.tools import tool

from aug.utils.data import read_data_file, write_data_file

_MEMORIES_FILE = "memories.json"


@tool
def remember(topic: str, content: str) -> str:
    """Save something to long-term memory.

    IMPORTANT: Before calling this, check the memory index in the system prompt.
    If a memory with a matching or closely related topic already exists, call
    update_memory instead — do not create duplicates.

    Args:
        topic: Short category label shown in every prompt (e.g. "Hotel preferences").
        content: Full content to store. Be thorough — include all relevant details.
    """
    memories = _load()
    memory = {"id": str(uuid.uuid4())[:8], "topic": topic, "content": content}
    memories.append(memory)
    _save(memories)
    return f"Memory saved with id={memory['id']}."


@tool
def recall(id: str) -> str:
    """Retrieve the full content of a memory by its id.

    Use this when a memory topic suggests it contains information relevant
    to the current conversation.

    Args:
        id: The memory id (shown in the system prompt index).
    """
    for m in _load():
        if m["id"] == id:
            return m["content"]
    return f"No memory found with id={id}."


@tool
def update_memory(id: str, content: str, topic: str | None = None) -> str:
    """Update the content of an existing memory, optionally renaming its topic.

    Use this to merge new information into an existing memory rather than
    creating a duplicate. Recall the memory first to read its current content,
    then write the merged result back.

    Args:
        id: The memory id to update.
        content: New content to replace the existing content with.
        topic: Optional new topic label. Omit to keep the existing one.
    """
    memories = _load()
    for m in memories:
        if m["id"] == id:
            m["content"] = content
            if topic is not None:
                m["topic"] = topic
            _save(memories)
            return f"Memory {id} updated."
    return f"No memory found with id={id}."


@tool
def forget(id: str) -> str:
    """Delete a memory permanently.

    Use this to remove outdated or incorrect memories.

    Args:
        id: The memory id to delete.
    """
    memories = _load()
    remaining = [m for m in memories if m["id"] != id]
    if len(remaining) == len(memories):
        return f"No memory found with id={id}."
    _save(remaining)
    return f"Memory {id} deleted."


def get_memory_index() -> str:
    """Return a formatted index of all memories for injection into system prompt."""
    memories = _load()
    if not memories:
        return ""
    lines = ["## Memories"] + [f"- [{m['id']}] {m['topic']}" for m in memories]
    return "\n".join(lines)


def _load() -> list[dict]:
    text = read_data_file(_MEMORIES_FILE)
    return json.loads(text) if text else []


def _save(memories: list[dict]) -> None:
    write_data_file(_MEMORIES_FILE, json.dumps(memories, indent=2))
