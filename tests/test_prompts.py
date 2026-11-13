"""build_session_prompt: what goes into a session's system prompt, and in what order."""

from aug.core.prompts import INTERFACE_PROMPTS, build_session_prompt


def _build(**overrides) -> str:
    args = {
        "workspace": "/ws",
        "agents_md": "AGENTS BODY",
        "interface": "telegram",
        "memory_index": "- user.md: who the user is",
        "skills": [("homelab", "Run the home server")],
    }
    return build_session_prompt(**{**args, **overrides})


def test_sections_come_in_a_fixed_order() -> None:
    prompt = _build()

    tags = ["<environment>", "<agents_md>", "<interface>", "<memory_index>", "<skills>"]
    positions = [prompt.index(tag) for tag in tags]
    assert positions == sorted(positions)


def test_environment_names_the_workspace_paths() -> None:
    prompt = _build(workspace="/app/data/workspace")

    assert "/app/data/workspace/memory" in prompt
    assert "/app/data/workspace/AGENTS.md" in prompt
    assert "/app/data/workspace/files/inbox" in prompt


def test_each_part_lands_in_the_prompt() -> None:
    prompt = _build()

    assert "AGENTS BODY" in prompt
    assert "- user.md: who the user is" in prompt
    assert "- homelab: Run the home server" in prompt
    assert INTERFACE_PROMPTS["telegram"].response_format.splitlines()[0] in prompt


def test_empty_parts_are_left_out() -> None:
    prompt = _build(agents_md="", memory_index="", skills=[], interface="unknown")

    for tag in ("<agents_md>", "<memory_index>", "<skills>", "<interface>"):
        assert tag not in prompt
    assert "<environment>" in prompt


def test_image_note_only_for_agents_that_describe_images() -> None:
    assert "describe_image" not in _build()
    assert "describe_image" in _build(describes_images=True)


def test_telegram_states_the_length_limit_not_a_style() -> None:
    context = INTERFACE_PROMPTS["telegram"].interface_context

    assert "4096" in context
    assert "concise" not in context.lower()


def test_same_inputs_give_the_same_bytes() -> None:
    assert _build() == _build()
