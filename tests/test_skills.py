"""Unit tests for the skills loader and skills tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from aug.utils.skills import (
    ALWAYS_ON_MAX_CHARS,
    Skill,
    SkillsIndex,
    build_skills_prompt,
    load_skill,
    load_skills,
    set_skill_name,
    validate_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(
    skills_dir: Path, name: str, description: str, body: str, always_on: bool = False
) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    metadata_line = "\nmetadata:\n  always_on: 'true'" if always_on else ""
    content = f"---\nname: {name}\ndescription: {description}{metadata_line}\n---\n\n{body}\n"
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------


def test_validate_name_valid():
    assert validate_name("my-skill") is None
    assert validate_name("skill123") is None
    assert validate_name("a") is None


def test_validate_name_empty():
    assert validate_name("") is not None


def test_validate_name_too_long():
    assert validate_name("a" * 65) is not None


def test_validate_name_uppercase():
    assert validate_name("My-Skill") is not None


def test_validate_name_leading_hyphen():
    assert validate_name("-skill") is not None


def test_validate_name_trailing_hyphen():
    assert validate_name("skill-") is not None


def test_validate_name_consecutive_hyphens():
    assert validate_name("my--skill") is not None


def test_validate_name_underscore():
    assert validate_name("my_skill") is not None


# ---------------------------------------------------------------------------
# load_skills
# ---------------------------------------------------------------------------


def test_load_skills_empty_dir(tmp_path):
    with patch("aug.utils.skills.SKILLS_DIR", tmp_path / "skills"):
        index = load_skills()
    assert index.always_on == []
    assert index.on_demand == []


def test_load_skills_missing_dir(tmp_path):
    with patch("aug.utils.skills.SKILLS_DIR", tmp_path / "nonexistent"):
        index = load_skills()
    assert index.always_on == []
    assert index.on_demand == []


def test_load_skills_on_demand(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "my-skill", "Does X when Y", "Follow these rules.")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert len(index.on_demand) == 1
    assert index.on_demand[0].name == "my-skill"
    assert index.on_demand[0].always_on is False
    assert index.always_on == []


def test_load_skills_always_on(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "style", "Response style", "Be concise.", always_on=True)
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert len(index.always_on) == 1
    assert index.always_on[0].always_on is True
    assert index.on_demand == []


def test_load_skills_skips_name_mismatch(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "wrong-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: right-name\ndescription: test\n---\n\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index.on_demand == []


def test_load_skills_skips_missing_description(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "no-desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index.on_demand == []


def test_load_skills_skips_missing_skill_md(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "empty-skill").mkdir(parents=True)
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index.on_demand == []


def test_load_skills_skips_malformed_yaml(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "bad-yaml"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\n: :\n---\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index.on_demand == []


# ---------------------------------------------------------------------------
# set_skill_name
# ---------------------------------------------------------------------------


def test_set_skill_name_rewrites_mismatched_name(tmp_path):
    # ClawHub slug "git2" ships a SKILL.md with name: git — the loader would reject it.
    skill_dir = tmp_path / "git2"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: git\ndescription: Git helper\n---\n\nDo git things.\n"
    )
    set_skill_name(skill_dir, "git2")
    detail = load_skill("git2", tmp_path)
    assert detail is not None
    assert detail.name == "git2"
    assert detail.description == "Git helper"
    assert detail.body == "Do git things."


def test_set_skill_name_preserves_other_frontmatter_and_body(tmp_path):
    skill_dir = tmp_path / "renamed"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: original\n"
        "description: Keeps everything\n"
        "metadata:\n"
        "  openclaw:\n"
        "    emoji: bee\n"
        "    homepage: https://example.com\n"
        "---\n\n"
        "Body line one.\n\nBody line two.\n"
    )
    set_skill_name(skill_dir, "renamed")
    content = (skill_dir / "SKILL.md").read_text()
    assert "name: renamed" in content
    assert "name: original" not in content
    assert "emoji: bee" in content
    assert "homepage: https://example.com" in content
    assert "Body line one." in content
    assert "Body line two." in content


def test_set_skill_name_idempotent_when_already_matching(tmp_path):
    skill_dir = tmp_path / "match"
    skill_dir.mkdir()
    original = "---\nname: match\ndescription: Same\n---\n\nbody\n"
    (skill_dir / "SKILL.md").write_text(original)
    set_skill_name(skill_dir, "match")
    detail = load_skill("match", tmp_path)
    assert detail is not None
    assert detail.name == "match"


def test_set_skill_name_inserts_name_when_absent(tmp_path):
    skill_dir = tmp_path / "noname"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\ndescription: Missing name\n---\n\nbody\n")
    set_skill_name(skill_dir, "noname")
    detail = load_skill("noname", tmp_path)
    assert detail is not None
    assert detail.name == "noname"
    assert detail.description == "Missing name"


def test_set_skill_name_handles_no_frontmatter(tmp_path):
    # A file lacking frontmatter gets a minimal one prepended (still not loadable
    # without a description, but the name must be present and the body preserved).
    skill_dir = tmp_path / "raw"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Just a body, no frontmatter.\n")
    set_skill_name(skill_dir, "raw")
    content = (skill_dir / "SKILL.md").read_text()
    assert content.startswith("---\nname: raw\n---")
    assert "Just a body, no frontmatter." in content


def test_set_skill_name_noop_when_no_skill_md(tmp_path):
    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    # Should not raise when SKILL.md is absent.
    set_skill_name(skill_dir, "empty")
    assert not (skill_dir / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# build_skills_prompt
# ---------------------------------------------------------------------------


def test_build_skills_prompt_empty():
    assert build_skills_prompt(SkillsIndex()) == ""


def test_build_skills_prompt_on_demand_only():
    index = SkillsIndex(
        on_demand=[
            Skill("my-skill", "Does X when Y", "body"),
            Skill("other-skill", "Does Z when W", "body2"),
        ]
    )
    prompt = build_skills_prompt(index)
    assert "get_skill" in prompt
    assert "my-skill: Does X when Y" in prompt
    assert "other-skill: Does Z when W" in prompt
    assert "body" not in prompt  # body not exposed in index


def test_build_skills_prompt_always_on_injects_body():
    index = SkillsIndex(always_on=[Skill("style", "Response style", "Be concise.", always_on=True)])
    prompt = build_skills_prompt(index)
    assert "Be concise." in prompt
    assert "get_skill" not in prompt


def test_build_skills_prompt_mixed():
    index = SkillsIndex(
        always_on=[Skill("style", "Style guide", "Be terse.", always_on=True)],
        on_demand=[Skill("ha", "HA automation guide", "Follow these patterns.")],
    )
    prompt = build_skills_prompt(index)
    assert "Be terse." in prompt
    assert "ha: HA automation guide" in prompt
    assert "Follow these patterns." not in prompt


# ---------------------------------------------------------------------------
# save_skill tool
# ---------------------------------------------------------------------------


def test_save_skill_creates_file(tmp_path):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="test-skill", description="A test.", body="Do this.")

    skill_md = tmp_path / "test-skill" / "SKILL.md"
    assert skill_md.exists()
    assert "test-skill" in skill_md.read_text()


def test_save_skill_returns_file_attachment(tmp_path):
    from aug.core.tools.output import FileAttachment, ToolOutput
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text, output = save_skill.func(name="test-skill", description="A test.", body="Do this.")

    assert isinstance(output, ToolOutput)
    assert len(output.attachments) == 1
    attachment = output.attachments[0]
    assert isinstance(attachment, FileAttachment)
    assert attachment.filename == "SKILL.md"
    assert b"test-skill" in attachment.data
    assert b"Do this." in attachment.data
    assert "test-skill" in text


def test_save_skill_invalid_name(tmp_path):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text, _ = save_skill.func(name="Bad_Name", description="test", body="body")

    assert "Bad_Name" not in [p.name for p in tmp_path.iterdir()]
    assert "name" in text.lower() or "invalid" in text.lower() or "lowercase" in text.lower()


def test_save_skill_always_on_size_enforcement(tmp_path):
    from aug.core.tools.skills import save_skill

    oversized_body = "x" * (ALWAYS_ON_MAX_CHARS + 1)
    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text, _ = save_skill.func(
            name="big-skill", description="test", body=oversized_body, always_on=True
        )

    assert "too large" in text
    assert str(ALWAYS_ON_MAX_CHARS) in text
    assert not (tmp_path / "big-skill").exists()


def test_save_skill_always_on_within_limit(tmp_path):
    from aug.core.tools.skills import save_skill

    body = "x" * ALWAYS_ON_MAX_CHARS
    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="small-skill", description="test", body=body, always_on=True)

    assert (tmp_path / "small-skill" / "SKILL.md").exists()
    assert "always_on" in (tmp_path / "small-skill" / "SKILL.md").read_text()


def test_save_skill_overwrites_existing(tmp_path):
    from aug.core.tools.output import ToolOutput
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="test-skill", description="v1", body="old body")
        _, output = save_skill.func(name="test-skill", description="v2", body="new body")

    content = (tmp_path / "test-skill" / "SKILL.md").read_text()
    assert "new body" in content
    assert "old body" not in content
    assert isinstance(output, ToolOutput)
    assert b"new body" in output.attachments[0].data


# ---------------------------------------------------------------------------
# write_skill_file tool
# ---------------------------------------------------------------------------


def test_write_skill_file_creates_script(tmp_path):
    from aug.core.tools.output import FileAttachment, ToolOutput
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="test-skill", description="test", body="body")
        text, output = write_skill_file.func(
            skill_name="test-skill", path="scripts/run.sh", content="echo hello"
        )

    script = tmp_path / "test-skill" / "scripts" / "run.sh"
    assert script.exists()
    assert script.read_text() == "echo hello"
    assert isinstance(output, ToolOutput)
    assert len(output.attachments) == 1
    attachment = output.attachments[0]
    assert isinstance(attachment, FileAttachment)
    assert attachment.filename == "run.sh"
    assert attachment.data == b"echo hello"
    assert "test-skill" in text


def test_write_skill_file_rejects_skill_md(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="test-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="test-skill", path="SKILL.md", content="overwrite"
        )

    assert "save_skill" in text


def test_write_skill_file_rejects_path_traversal(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="test-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="test-skill", path="../other-skill/SKILL.md", content="evil"
        )

    assert "escapes" in text
    assert not (tmp_path / "other-skill").exists()


def test_write_skill_file_skill_not_found(tmp_path):
    from aug.core.tools.skills import write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text, _ = write_skill_file.func(
            skill_name="nonexistent", path="scripts/run.sh", content="echo hi"
        )

    assert "not found" in text


def test_write_skill_file_with_dependencies_injects_and_installs(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        save_skill.func(name="dep-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="dep-skill",
            path="scripts/x.py",
            content="import httpx\n",
            dependencies=["httpx>=0.27"],
        )

    written = (tmp_path / "dep-skill" / "scripts" / "x.py").read_text()
    assert "# /// script" in written  # PEP 723 block was written for the agent
    assert "httpx>=0.27" in written
    assert "import httpx" in written
    assert "installed and cached: httpx>=0.27" in text
    assert "uv run scripts/x.py" in text
    mock_run.assert_called_once()


def test_write_skill_file_reports_dependency_install_failure(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no match for 'nope'")
        save_skill.func(name="dep-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="dep-skill",
            path="scripts/x.py",
            content="import nope\n",
            dependencies=["nope"],
        )

    # The file is still written, but the agent is told the install failed.
    assert (tmp_path / "dep-skill" / "scripts" / "x.py").exists()
    assert "FAILED" in text
    assert "no match for 'nope'" in text


def test_write_skill_file_rejects_handwritten_pep723_block(tmp_path):
    # Hand-writing the PEP 723 block instead of using `dependencies` is rejected, steering
    # the agent to the one supported path.
    from aug.core.tools.skills import save_skill, write_skill_file

    content = '# /// script\n# dependencies = ["httpx"]\n# ///\nimport httpx\n'
    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="dep-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="dep-skill", path="scripts/x.py", content=content
        )

    assert "dependencies" in text and "parameter" in text
    assert not (tmp_path / "dep-skill" / "scripts" / "x.py").exists()


def test_write_skill_file_rejects_dependencies_for_non_python(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.func(name="dep-skill", description="test", body="body")
        text, _ = write_skill_file.func(
            skill_name="dep-skill",
            path="scripts/run.sh",
            content="echo hi",
            dependencies=["httpx"],
        )

    assert "only supported for Python" in text
    assert not (tmp_path / "dep-skill" / "scripts" / "run.sh").exists()


# ---------------------------------------------------------------------------
# get_skill tool
# ---------------------------------------------------------------------------


def test_get_skill_returns_content(tmp_path):
    from aug.core.tools.skills import get_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.invoke({"name": "my-skill", "description": "test", "body": "Follow this."})
        result = get_skill.invoke({"name": "my-skill"})

    assert "Follow this." in result


def test_get_skill_surfaces_bundled_script_dependencies(tmp_path):
    from aug.core.tools.skills import get_skill, save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        save_skill.func(name="my-skill", description="test", body="Follow this.")
        write_skill_file.func(
            skill_name="my-skill",
            path="scripts/x.py",
            content="import httpx\n",
            dependencies=["httpx>=0.27"],
        )
        result = get_skill.invoke({"name": "my-skill"})

    assert "Bundled scripts" in result
    assert "scripts/x.py" in result
    assert "httpx>=0.27" in result


def test_get_skill_not_found(tmp_path):
    from aug.core.tools.skills import get_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        result = get_skill.invoke({"name": "nonexistent"})

    assert "not found" in result


# ---------------------------------------------------------------------------
# delete_skill tool
# ---------------------------------------------------------------------------


def test_delete_skill_removes_directory(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.invoke({"name": "test-skill", "description": "test", "body": "body"})
        result = delete_skill.invoke({"skill_name": "test-skill"})

    assert not (tmp_path / "test-skill").exists()
    assert "deleted" in result.lower()


def test_delete_skill_removes_single_file(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.invoke({"name": "test-skill", "description": "test", "body": "body"})
        write_skill_file.invoke(
            {"skill_name": "test-skill", "path": "scripts/run.sh", "content": "echo hi"}
        )
        result = delete_skill.invoke({"skill_name": "test-skill", "path": "scripts/run.sh"})

    assert not (tmp_path / "test-skill" / "scripts" / "run.sh").exists()
    assert (tmp_path / "test-skill" / "SKILL.md").exists()  # SKILL.md untouched
    assert "Deleted" in result


def test_delete_skill_not_found(tmp_path):
    from aug.core.tools.skills import delete_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        result = delete_skill.invoke({"skill_name": "nonexistent"})

    assert "not found" in result


def test_delete_skill_file_path_traversal(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.invoke({"name": "test-skill", "description": "test", "body": "body"})
        result = delete_skill.invoke({"skill_name": "test-skill", "path": "../other/SKILL.md"})

    assert "escapes" in result


def test_delete_skill_prevents_skill_md_deletion(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        save_skill.invoke({"name": "test-skill", "description": "test", "body": "body"})
        result = delete_skill.invoke({"skill_name": "test-skill", "path": "SKILL.md"})

    assert (tmp_path / "test-skill" / "SKILL.md").exists()
    assert "Cannot delete SKILL.md" in result


def test_parse_frontmatter_with_dashes_in_value(tmp_path):
    """Frontmatter values containing '---' must not confuse the parser."""
    from aug.utils.skills import _parse_skill_md

    raw = "---\nname: my-skill\ndescription: Use --- to separate things\n---\n\nbody content\n"
    frontmatter, body = _parse_skill_md(raw)
    assert frontmatter["name"] == "my-skill"
    assert "body content" in body
