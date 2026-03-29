"""Skills tools — create, read, update and delete agentskills.io-compatible skill files."""

import shutil
from pathlib import Path

import yaml
from langchain_core.tools import tool

from aug.core.tools.output import FileAttachment, ToolOutput
from aug.utils.skills import ALWAYS_ON_MAX_CHARS, SKILLS_DIR, validate_name


@tool
def get_skill(name: str) -> str:
    """Load the full instructions for a named skill.

    Returns the complete SKILL.md content. Call this when the skills index lists a skill
    that is relevant to the current task.

    Args:
        name: Skill name as shown in the index (e.g. 'ha-automations').
    """
    skill_dir = SKILLS_DIR / name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return f"Skill '{name}' not found."
    return skill_md.read_text()


@tool(response_format="content_and_artifact")
def save_skill(
    name: str,
    description: str,
    body: str,
    always_on: bool = False,
) -> tuple[str, ToolOutput]:
    """Create or overwrite a skill's SKILL.md.

    On success: returns the written SKILL.md as a file attachment. Pass it through to the user
    as-is — do not summarise or paraphrase. The user verifies the file directly.
    On failure: returns an error string. Do not proceed if an error is returned.

    Args:
        name: Skill name. Lowercase letters, numbers, and hyphens only. Max 64 chars.
              Must not start or end with a hyphen or contain consecutive hyphens.
        description: What this skill does and when to use it. Max 1024 chars.
        body: Markdown instructions. The main content of the skill.
        always_on: If True, full content is injected into every system prompt.
                   Body must be ≤ 1000 chars. Use for short, universally relevant skills only.
    """
    if err := validate_name(name):
        return err, ToolOutput(text=err)
    if not description:
        msg = "Description must not be empty."
        return msg, ToolOutput(text=msg)
    if len(description) > 1024:
        msg = f"Description must be ≤ 1024 characters (got {len(description)})."
        return msg, ToolOutput(text=msg)
    if always_on and len(body) > ALWAYS_ON_MAX_CHARS:
        msg = (
            f"Body is too large for always_on ({len(body)} chars). "
            f"Shorten to {ALWAYS_ON_MAX_CHARS} chars or fewer, or pass always_on=False."
        )
        return msg, ToolOutput(text=msg)

    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    frontmatter: dict = {"name": name, "description": description}
    if always_on:
        frontmatter["metadata"] = {"always_on": "true"}

    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False).strip()}\n---\n\n{body}\n"
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content)
    output = ToolOutput(
        text=f"Skill '{name}' saved.",
        attachments=[FileAttachment(data=content.encode(), filename="SKILL.md")],
    )
    return f"Skill '{name}' saved.", output


@tool(response_format="content_and_artifact")
def write_skill_file(skill_name: str, path: str, content: str) -> tuple[str, ToolOutput]:
    """Add or update a file inside an existing skill directory (scripts, references, assets).

    Do not use this for SKILL.md — use save_skill instead.
    On success: returns the written file as an attachment. Pass it through to the user
    as-is — do not summarise or paraphrase.
    On failure: returns an error string.

    Args:
        skill_name: Name of the skill directory to write into.
        path: Relative path within the skill directory, e.g. 'scripts/check.py'.
              Must not be 'SKILL.md' and must not escape the skill directory.
        content: File content to write.
    """
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.exists():
        msg = f"Skill '{skill_name}' not found. Create it with save_skill first."
        return msg, ToolOutput(text=msg)

    # Resolve and validate path stays inside skill_dir
    try:
        target = (skill_dir / path).resolve()
        skill_dir_resolved = skill_dir.resolve()
        target.relative_to(skill_dir_resolved)  # raises ValueError if outside
    except ValueError:
        msg = f"Path '{path}' escapes the skill directory — not allowed."
        return msg, ToolOutput(text=msg)

    if target == (skill_dir / "SKILL.md").resolve():
        msg = "Use save_skill to update SKILL.md."
        return msg, ToolOutput(text=msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    text = f"Written '{path}' to skill '{skill_name}'."
    output = ToolOutput(
        text=text,
        attachments=[FileAttachment(data=content.encode(), filename=Path(path).name)],
    )
    return text, output


@tool
def delete_skill(skill_name: str, path: str | None = None) -> str:
    """Delete a skill or a specific file within a skill.

    Args:
        skill_name: Name of the skill to delete or modify.
        path: If omitted, deletes the entire skill directory.
              If provided, deletes only that file (e.g. 'scripts/old.py').
    """
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.exists():
        return f"Skill '{skill_name}' not found."

    if path is None:
        shutil.rmtree(skill_dir)
        return f"Skill '{skill_name}' deleted."

    try:
        target = (skill_dir / path).resolve()
        skill_dir.resolve()
        target.relative_to(skill_dir.resolve())  # raises ValueError if outside
    except ValueError:
        return f"Path '{path}' escapes the skill directory — not allowed."

    if target == (skill_dir / "SKILL.md").resolve():
        return (
            "Cannot delete SKILL.md directly. "
            "Use delete_skill without a path to remove the entire skill."
        )

    if not target.exists():
        return f"File '{path}' not found in skill '{skill_name}'."

    target.unlink()
    return f"Deleted '{path}' from skill '{skill_name}'."
