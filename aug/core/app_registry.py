"""Minimal FastAPI app reference, shared by modules that must reach
app.state without importing aug.core.dispatch.

aug.core.registry imports aug.core.tools.mcp, and aug.core.dispatch imports
aug.core.registry (for get_agent) — so tools/mcp.py importing dispatch.py
directly would form a cycle (dispatch -> registry -> tools.mcp -> dispatch).
This module has no such dependents, so it can sit underneath both.
"""

from fastapi import FastAPI

_app: FastAPI | None = None


def set_app(app: FastAPI) -> None:
    global _app
    _app = app


def get_app() -> FastAPI | None:
    """Return the live FastAPI app, or None before startup has wired one in."""
    return _app
