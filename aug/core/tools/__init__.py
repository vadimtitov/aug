"""AUG tools — LangChain @tool-decorated functions.

To add a new tool:
1. Create ``aug/core/tools/<name>.py`` and decorate your function with ``@tool``.
2. Import it in the relevant agent and pass it to ``build_llm(...).bind_tools([...])``.
"""
