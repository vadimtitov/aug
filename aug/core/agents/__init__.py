"""Agent implementations.

Each agent module contains a class that inherits ``BaseAgent`` and overrides:
    AGENT_CONFIG (dict)      — model + generation params
    build_graph() -> StateGraph  — define nodes and edges (no .compile() call)
"""
