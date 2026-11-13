"""Test configuration — set required env vars before any aug module is imported."""

import os

import pytest

# These must be set at module level (before pytest collects tests) so that
# aug.config.Settings() can be instantiated without a real .env file.
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:4000")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("BASE_URL", "https://aug.test")
os.environ.setdefault("OAUTH_ENCRYPTION_KEY", "A" * 43 + "=")  # 32 zero-ish bytes, base64


# aug imports only after the env vars above are set.
from aug.core.harness import Address, ToolContext, TurnContext
from aug.utils.storage import file_store


@pytest.fixture(autouse=True)
def _workspace_files(tmp_path, monkeypatch):
    """Every test stores its files in its own tmp dir, never under /app/data."""
    monkeypatch.setattr(file_store, "root", tmp_path / "files")


@pytest.fixture
def sent_files(monkeypatch) -> list:
    """Files AUG's tools send to the user during the test, as FileEmitted events.

    The tools see a running turn's ToolContext, whose emit records into the list.
    """
    sent: list = []

    async def emit(event) -> None:
        sent.append(event)

    ctx = ToolContext(
        context_id="test-context",
        turn=TurnContext(
            reply_to=Address(interface="telegram", conversation_id="c"), attended=True
        ),
        _emit=emit,
    )
    for module in (
        "aug.core.tools.respond_with_file",
        "aug.core.tools.image_gen",
        "aug.core.tools.browser",
    ):
        monkeypatch.setattr(f"{module}.current_tool_context", lambda: ctx)
    return sent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace of the test's own (tmp/workspace), with the pre-workspace folders at
    tmp/legacy/memory and tmp/legacy/skills.  Returns the workspace root."""
    root = tmp_path / "workspace"
    legacy = tmp_path / "legacy"
    paths = {
        "WORKSPACE_DIR": root,
        "AGENTS_MD": root / "AGENTS.md",
        "MEMORY_DIR": root / "memory",
        "MEMORY_INDEX": root / "memory" / "MEMORY.md",
        "SKILLS_DIR": root / "skills",
    }
    for name, path in paths.items():
        monkeypatch.setattr(f"aug.core.workspace.{name}", path)
    monkeypatch.setattr("aug.core.workspace._LEGACY_MEMORY_DIR", legacy / "memory")
    monkeypatch.setattr("aug.core.workspace._LEGACY_SKILLS_DIR", legacy / "skills")
    monkeypatch.setattr("aug.core.memory.MEMORY_DIR", paths["MEMORY_DIR"])
    monkeypatch.setattr("aug.core.memory.MEMORY_INDEX", paths["MEMORY_INDEX"])
    monkeypatch.setattr("aug.core.tools.skills.SKILLS_DIR", paths["SKILLS_DIR"])
    monkeypatch.setattr("aug.utils.skills.SKILLS_DIR", paths["SKILLS_DIR"])
    return root
