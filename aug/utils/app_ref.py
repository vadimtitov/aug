"""FastAPI app singleton, for modules that need app.state without importing
aug.core.dispatch and risking an import cycle (dispatch -> registry -> a tool
module -> dispatch). Same pattern as get_pool()/set_pool() in aug/utils/db.py.
"""

from fastapi import FastAPI

_app: FastAPI | None = None


def set_app(app: FastAPI) -> None:
    global _app
    _app = app


def get_app() -> FastAPI | None:
    """The live FastAPI app, or None before startup has wired one in."""
    return _app
