"""Skills tools — create, read, update and delete agentskills.io-compatible skill files."""

import logging
import shutil
from pathlib import Path

from langchain_core.tools import tool

from aug.core.skill_deps import (
    find_pep723_scripts,
    has_pep723_block,
    inject_dependencies,
    read_pep723_deps,
    resolve_dependencies,
)
from aug.core.tools.output import FileAttachment, ToolOutput
from aug.utils.skills import ALWAYS_ON_MAX_CHARS, SKILLS_DIR, validate_name, write_skill_md

logger = logging.getLogger(__name__)


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
    return skill_md.read_text() + _bundled_scripts_note(skill_dir)


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
    write_skill_md(skill_dir, name, description, body, always_on)
    content = (skill_dir / "SKILL.md").read_text()
    output = ToolOutput(
        text=f"Skill '{name}' saved.",
        attachments=[FileAttachment(data=content.encode(), filename="SKILL.md")],
    )
    return f"Skill '{name}' saved.", output


@tool(response_format="content_and_artifact")
def write_skill_file(
    skill_name: str,
    path: str,
    content: str,
    dependencies: list[str] | None = None,
) -> tuple[str, ToolOutput]:
    """Add or update a file inside an existing skill directory (scripts, references, assets).

    Do not use this for SKILL.md — use save_skill instead.
    On success: returns the written file as an attachment. Pass it through to the user
    as-is — do not summarise or paraphrase.
    On failure: returns an error string.

    Dependencies (Python scripts): when a `.py` script imports any third-party package, you
    MUST pass those packages in `dependencies` — e.g. dependencies=["pdfplumber>=0.11"]. Do
    NOT hand-write a `# /// script` / PEP 723 block in `content`: the tool writes that block
    for you, installs the packages immediately into an isolated cached environment, and tells
    you in the return value whether the install succeeded (fix and call again if a name was
    wrong). The script's `content` should contain only the code (imports + logic), no
    dependency metadata. The skill's instructions must run the script with `uv run <path>`,
    never `python <path>`. Pin versions for stability. For Node tools, call `npx pkg@version`
    from the instructions instead (no `dependencies` needed).

    Args:
        skill_name: Name of the skill directory to write into.
        path: Relative path within the skill directory, e.g. 'scripts/check.py'.
              Must not be 'SKILL.md' and must not escape the skill directory.
        content: File content — code only. Do not include a PEP 723 (`# /// script`) block;
                 declare packages via `dependencies` instead.
        dependencies: Third-party Python packages the script imports (e.g. ["httpx>=0.27"]).
                      Only valid for '.py' files. Installed immediately; result reported back.
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

    if dependencies and target.suffix != ".py":
        msg = f"dependencies are only supported for Python (.py) scripts, not '{path}'."
        return msg, ToolOutput(text=msg)

    # Enforce one obvious path: declared dependencies always go through the `dependencies`
    # parameter, never a hand-written PEP 723 block. This guarantees the tool owns the block
    # (so it's always valid) and that the packages are actually installed and verified.
    if dependencies is None and target.suffix == ".py" and has_pep723_block(content):
        logger.info(
            "write_skill_file steering hand-written PEP 723 to dependencies param path=%s", path
        )
        msg = (
            "Don't hand-write the `# /// script` (PEP 723) block. Remove it from `content` and "
            "pass the packages in the `dependencies` parameter instead — the tool writes the "
            "block and installs them for you. Call write_skill_file again."
        )
        return msg, ToolOutput(text=msg)

    if dependencies is not None:
        content = inject_dependencies(content, dependencies)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    text = _write_message(skill_name, path, target)
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


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


def _write_message(skill_name: str, path: str, target: Path) -> str:
    """Build the result string for write_skill_file, installing declared deps if any."""
    deps = read_pep723_deps(target)
    if not deps:
        return f"Written '{path}' to skill '{skill_name}'."

    result = resolve_dependencies(target)
    if result.ok:
        return (
            f"Wrote '{path}'. Dependencies installed and cached: {', '.join(deps)}. "
            f"Run the script with `uv run {path}`."
        )
    return (
        f"Wrote '{path}', but installing its dependencies ({', '.join(deps)}) FAILED:\n"
        f"{result.detail}\n"
        "Fix the package names or versions and call write_skill_file again."
    )


def _bundled_scripts_note(skill_dir: Path) -> str:
    """A short awareness note listing each bundled script's declared dependencies."""
    lines: list[str] = []
    for script in find_pep723_scripts(skill_dir):
        rel = script.relative_to(skill_dir)
        deps = read_pep723_deps(script)
        lines.append(f"- {rel} — {', '.join(deps) if deps else 'no third-party packages'}")
    if not lines:
        return ""
    body = "\n".join(lines)
    return "\n\n---\nBundled scripts (dependencies auto-install on `uv run <path>`):\n" + body
