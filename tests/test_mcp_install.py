"""Tests for aug/core/mcp_install.py — MCP install-plan resolution/building."""

from unittest.mock import patch

import pytest

from aug.core import mcp_install
from aug.utils.mcp_registry import McpRegistryServer
from aug.utils.state import AppState, McpCredentialBinding, McpInstallPlan


def _server(
    name="io.github.x/server-postgres", transport="stdio", required_inputs=None
) -> McpRegistryServer:
    return McpRegistryServer(
        name=name,
        namespace="io.github.x",
        description="A test server",
        version="1.0.0",
        transport=transport,
        command="npx",
        args=["-y", "server-postgres@1.0.0"],
        url="https://example.com/mcp" if transport == "http" else "",
        required_inputs=required_inputs or [],
    )


# ---------------------------------------------------------------------------
# get_or_build_plan — approval interrupt/resume persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_build_plan_reuses_persisted_plan_across_a_different_search():
    """An in-progress install must not be resolved against a different, later
    search snapshot — a persisted plan takes priority over whatever snapshot
    the current call is passed."""
    snapshot_a = [_server(name="io.github.x/server-postgres")]
    state = AppState()

    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.mcp_install.list_secret_names", return_value=set()),
    ):
        plan = await mcp_install.get_or_build_plan("thread-a", 1, snapshot_a)
        assert plan is not None
        assert plan.slug == "postgres-0650d2"

        # A later, different snapshot for the same thread (LangGraph replaying
        # the node on resume) must not shadow the already-persisted plan.
        snapshot_b = [_server(name="io.github.x/server-mysql")]
        again = await mcp_install.get_or_build_plan("thread-a", 1, snapshot_b)
    assert again == plan


@pytest.mark.asyncio
async def test_get_or_build_plan_survives_missing_live_snapshot():
    """Simulates a process restart while approval was pending: the in-memory
    search cache is gone, but the persisted plan (state.json) is not."""
    plan = McpInstallPlan(
        id="p1",
        thread_id="thread-a",
        search_index=1,
        server_name="io.github.x/server-postgres",
        slug="postgres",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "server-postgres@1.0.0"],
        credentials=[McpCredentialBinding(target_name="X", secret_name="X", bound=True)],
    )
    state = AppState()
    state.mcp.install_plans["thread-a"] = plan

    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state"),
        patch("aug.core.mcp_install.list_secret_names", return_value={"X"}),
    ):
        resolved = await mcp_install.get_or_build_plan("thread-a", 1, [])

    assert resolved == plan


@pytest.mark.asyncio
async def test_get_or_build_plan_revalidates_bound_status_on_retry():
    """A plan built while a secret was missing must notice once the user adds
    it and retries the same index, rather than reusing the stale persisted
    plan forever."""
    plan = McpInstallPlan(
        id="p1",
        thread_id="thread-a",
        search_index=1,
        server_name="io.github.x/server-github",
        slug="github",
        version="1.0.0",
        transport="stdio",
        command="npx",
        args=["-y", "server-github@1.0.0"],
        credentials=[
            McpCredentialBinding(
                target_name="GITHUB_TOKEN", secret_name="GITHUB_TOKEN", bound=False
            )
        ],
    )
    state = AppState()
    state.mcp.install_plans["thread-a"] = plan
    saved = []

    with (
        patch("aug.utils.state.load_state", return_value=state),
        patch("aug.utils.state.save_state", side_effect=lambda s: saved.append(s)),
        patch("aug.core.mcp_install.list_secret_names", return_value={"GITHUB_TOKEN"}),
    ):
        resolved = await mcp_install.get_or_build_plan("thread-a", 1, [])

    assert resolved.credentials[0].bound is True
    assert len(saved) == 1
    assert saved[0].mcp.install_plans["thread-a"].credentials[0].bound is True


# ---------------------------------------------------------------------------
# build_plan / plan_to_config / secret_name_for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_plan_preserves_literal_inputs():
    """Registry-declared literal/default values must survive into the
    persisted plan (and from there into the saved config's env_static)."""
    server = _server(name="io.github.x/server-simple")
    server = server.model_copy(update={"literal_inputs": {"LOG_LEVEL": "info"}})

    plan = mcp_install.build_plan("thread-a", 1, server, known_secrets=set())

    assert plan.literal_inputs == {"LOG_LEVEL": "info"}
    cfg = mcp_install.plan_to_config(plan, bindings={})
    assert cfg.env_static == {"LOG_LEVEL": "info"}


@pytest.mark.parametrize(
    "target_name,expected",
    [
        ("DATABASE_URL", "DATABASE_URL"),
        ("X-API-Key", "X_API_KEY"),
        ("Authorization", "AUTHORIZATION"),
    ],
)
def test_secret_name_for_sanitizes(target_name, expected):
    assert mcp_install.secret_name_for(target_name) == expected
