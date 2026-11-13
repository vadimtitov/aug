"""Unit tests for the skills loader and skills tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from aug.utils.skills import (
    load_skill,
    load_skills,
    set_skill_name,
    validate_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(skills_dir: Path, name: str, description: str, body: str) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
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
    assert index == []


def test_load_skills_missing_dir(tmp_path):
    with patch("aug.utils.skills.SKILLS_DIR", tmp_path / "nonexistent"):
        index = load_skills()
    assert index == []


def test_load_skills_on_demand(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "my-skill", "Does X when Y", "Follow these rules.")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert [s.name for s in index] == ["my-skill"]


def test_load_skills_skips_name_mismatch(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "wrong-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: right-name\ndescription: test\n---\n\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index == []


def test_load_skills_skips_missing_description(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "no-desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index == []


def test_load_skills_skips_missing_skill_md(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "empty-skill").mkdir(parents=True)
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index == []


def test_load_skills_skips_malformed_yaml(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "bad-yaml"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\n: :\n---\nbody\n")
    with patch("aug.utils.skills.SKILLS_DIR", skills_dir):
        index = load_skills()
    assert index == []


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
# save_skill tool
# ---------------------------------------------------------------------------


async def test_save_skill_creates_file(tmp_path):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.coroutine(name="test-skill", description="A test.", body="Do this.")

    skill_md = tmp_path / "test-skill" / "SKILL.md"
    assert skill_md.exists()
    assert "test-skill" in skill_md.read_text()


async def test_save_skill_sends_skill_md(tmp_path, sent_files):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path / "skills"):
        text = await save_skill.coroutine(name="test-skill", description="A test.", body="Do this.")

    assert len(sent_files) == 1
    assert sent_files[0].filename == "SKILL.md"
    sent = sent_files[0].path.read_bytes()
    assert b"test-skill" in sent
    assert b"Do this." in sent
    assert "test-skill" in text


async def test_save_skill_invalid_name(tmp_path):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text = await save_skill.coroutine(name="Bad_Name", description="test", body="body")

    assert "Bad_Name" not in [p.name for p in tmp_path.iterdir()]
    assert "name" in text.lower() or "invalid" in text.lower() or "lowercase" in text.lower()


async def test_save_skill_overwrites_existing(tmp_path, sent_files):
    from aug.core.tools.skills import save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path / "skills"):
        await save_skill.coroutine(name="test-skill", description="v1", body="old body")
        await save_skill.coroutine(name="test-skill", description="v2", body="new body")

    content = (tmp_path / "skills" / "test-skill" / "SKILL.md").read_text()
    assert "new body" in content
    assert "old body" not in content
    # Each send is a snapshot: the first still holds what was sent then.
    assert b"old body" in sent_files[0].path.read_bytes()
    assert b"new body" in sent_files[1].path.read_bytes()


# ---------------------------------------------------------------------------
# write_skill_file tool
# ---------------------------------------------------------------------------


async def test_write_skill_file_creates_script(tmp_path, sent_files):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path / "skills"):
        await save_skill.coroutine(name="test-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
            skill_name="test-skill", path="scripts/run.sh", content="echo hello"
        )

    script = tmp_path / "skills" / "test-skill" / "scripts" / "run.sh"
    assert script.read_text() == "echo hello"
    assert [f.filename for f in sent_files] == ["SKILL.md", "run.sh"]
    assert sent_files[1].path.read_bytes() == b"echo hello"
    assert "test-skill" in text


async def test_write_skill_file_rejects_skill_md(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.coroutine(name="test-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
            skill_name="test-skill", path="SKILL.md", content="overwrite"
        )

    assert "save_skill" in text


async def test_write_skill_file_rejects_path_traversal(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.coroutine(name="test-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
            skill_name="test-skill", path="../other-skill/SKILL.md", content="evil"
        )

    assert "escapes" in text
    assert not (tmp_path / "other-skill").exists()


async def test_write_skill_file_skill_not_found(tmp_path):
    from aug.core.tools.skills import write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        text = await write_skill_file.coroutine(
            skill_name="nonexistent", path="scripts/run.sh", content="echo hi"
        )

    assert "not found" in text


async def test_write_skill_file_with_dependencies_injects_and_installs(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        await save_skill.coroutine(name="dep-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
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


async def test_write_skill_file_reports_dependency_install_failure(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no match for 'nope'")
        await save_skill.coroutine(name="dep-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
            skill_name="dep-skill",
            path="scripts/x.py",
            content="import nope\n",
            dependencies=["nope"],
        )

    # The file is still written, but the agent is told the install failed.
    assert (tmp_path / "dep-skill" / "scripts" / "x.py").exists()
    assert "FAILED" in text
    assert "no match for 'nope'" in text


async def test_write_skill_file_rejects_handwritten_pep723_block(tmp_path):
    # Hand-writing the PEP 723 block instead of using `dependencies` is rejected, steering
    # the agent to the one supported path.
    from aug.core.tools.skills import save_skill, write_skill_file

    content = '# /// script\n# dependencies = ["httpx"]\n# ///\nimport httpx\n'
    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.coroutine(name="dep-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
            skill_name="dep-skill", path="scripts/x.py", content=content
        )

    assert "dependencies" in text and "parameter" in text
    assert not (tmp_path / "dep-skill" / "scripts" / "x.py").exists()


async def test_write_skill_file_rejects_dependencies_for_non_python(tmp_path):
    from aug.core.tools.skills import save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.coroutine(name="dep-skill", description="test", body="body")
        text = await write_skill_file.coroutine(
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


async def test_get_skill_returns_content(tmp_path):
    from aug.core.tools.skills import get_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.ainvoke(
            {"name": "my-skill", "description": "test", "body": "Follow this."}
        )
        result = get_skill.invoke({"name": "my-skill"})

    assert "Follow this." in result


async def test_get_skill_surfaces_bundled_script_dependencies(tmp_path):
    from aug.core.tools.skills import get_skill, save_skill, write_skill_file

    with (
        patch("aug.core.tools.skills.SKILLS_DIR", tmp_path),
        patch("aug.core.skill_deps.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        await save_skill.coroutine(name="my-skill", description="test", body="Follow this.")
        await write_skill_file.coroutine(
            skill_name="my-skill",
            path="scripts/x.py",
            content="import httpx\n",
            dependencies=["httpx>=0.27"],
        )
        result = get_skill.invoke({"name": "my-skill"})

    assert "Bundled scripts" in result
    assert "scripts/x.py" in result
    assert "httpx>=0.27" in result


async def test_get_skill_not_found(tmp_path):
    from aug.core.tools.skills import get_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        result = get_skill.invoke({"name": "nonexistent"})

    assert "not found" in result


# ---------------------------------------------------------------------------
# delete_skill tool
# ---------------------------------------------------------------------------


async def test_delete_skill_removes_directory(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.ainvoke({"name": "test-skill", "description": "test", "body": "body"})
        result = delete_skill.invoke({"skill_name": "test-skill"})

    assert not (tmp_path / "test-skill").exists()
    assert "deleted" in result.lower()


async def test_delete_skill_removes_single_file(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill, write_skill_file

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.ainvoke({"name": "test-skill", "description": "test", "body": "body"})
        await write_skill_file.ainvoke(
            {"skill_name": "test-skill", "path": "scripts/run.sh", "content": "echo hi"}
        )
        result = delete_skill.invoke({"skill_name": "test-skill", "path": "scripts/run.sh"})

    assert not (tmp_path / "test-skill" / "scripts" / "run.sh").exists()
    assert (tmp_path / "test-skill" / "SKILL.md").exists()  # SKILL.md untouched
    assert "Deleted" in result


async def test_delete_skill_not_found(tmp_path):
    from aug.core.tools.skills import delete_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        result = delete_skill.invoke({"skill_name": "nonexistent"})

    assert "not found" in result


async def test_delete_skill_file_path_traversal(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.ainvoke({"name": "test-skill", "description": "test", "body": "body"})
        result = delete_skill.invoke({"skill_name": "test-skill", "path": "../other/SKILL.md"})

    assert "escapes" in result


async def test_delete_skill_prevents_skill_md_deletion(tmp_path):
    from aug.core.tools.skills import delete_skill, save_skill

    with patch("aug.core.tools.skills.SKILLS_DIR", tmp_path):
        await save_skill.ainvoke({"name": "test-skill", "description": "test", "body": "body"})
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
