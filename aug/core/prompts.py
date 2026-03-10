"""System prompts for each agent.

Keeping prompts in one place makes it easy to iterate on behaviour without
touching graph/agent logic.
"""

DEFAULT_SYSTEM_PROMPT = """\
You are AUG, a personal AI assistant. You are helpful, concise, and honest.
When you don't know something, say so rather than making things up.
"""

# Add new agent prompts here as the roster grows.
PROMPTS: dict[str, str] = {
    "default": DEFAULT_SYSTEM_PROMPT,
}


def get_prompt(agent: str) -> str:
    """Return the system prompt for *agent*, falling back to the default."""
    return PROMPTS.get(agent, DEFAULT_SYSTEM_PROMPT)
