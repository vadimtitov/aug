"""Test configuration — set required env vars before any aug module is imported."""

import os

# These must be set at module level (before pytest collects tests) so that
# aug.config.Settings() can be instantiated without a real .env file.
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:4000")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
