"""Skills loader — reads agentskills.io-compatible skill directories from workspace/skills/.

Each skill is a directory containing a SKILL.md file with YAML frontmatter.  The
session prompt lists every skill as one index line; the agent loads the rest on demand.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aug.utils.data import SKILLS_DIR

logger = logging.getLogger(__name__)


# Name validation per agentskills.io spec
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_CONSECUTIVE_HYPHENS_RE = re.compile(r"--")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


@dataclass(frozen=True)
class SkillDetail:
    name: str
    description: str
    body: str
    files: list[str]


def validate_name(name: str) -> str | None:
    """Return an error string if name is invalid, None if valid."""
    if not name:
        return "Skill name must not be empty."
    if len(name) > 64:
        return f"Skill name must be ≤ 64 characters (got {len(name)})."
    if _CONSECUTIVE_HYPHENS_RE.search(name):
        return "Skill name must not contain consecutive hyphens."
    if not _NAME_RE.match(name):
        return (
            "Skill name must contain only lowercase letters, numbers, and hyphens, "
            "and must not start or end with a hyphen."
        )
    return None


def load_skills(skills_dir: Path | None = None) -> list[Skill]:
    """Every skill in skills_dir (defaults to SKILLS_DIR), sorted by name.

    Malformed skill directories are skipped with a warning — never crash the prompt builder.
    """
    directory = skills_dir or SKILLS_DIR
    if not directory.exists():
        return []
    skills = (_load_skill(d) for d in sorted(directory.iterdir()) if d.is_dir())
    return [s for s in skills if s is not None]


def write_skill_md(skill_dir: Path, name: str, description: str, body: str) -> None:
    """Assemble and write SKILL.md from structured fields.  Any other frontmatter the
    file already has (license, allowed-tools, metadata…) is kept."""
    skill_md = skill_dir / "SKILL.md"
    existing: dict = {}
    if skill_md.exists():
        try:
            existing, _ = _parse_skill_md(skill_md.read_text())
        except yaml.YAMLError:
            logger.warning("write_skill_md %s: unreadable frontmatter replaced", skill_dir.name)
    frontmatter = {**existing, "name": name, "description": description}
    dumped = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True)
    content = f"---\n{dumped.strip()}\n---\n\n{body}\n"
    skill_md.write_text(content)


def set_skill_name(skill_dir: Path, name: str) -> None:
    """Force the top-level ``name`` in a skill's SKILL.md frontmatter to equal ``name``.

    ClawHub assigns its own slug (used here as the install directory name), which can
    differ from the author's frontmatter ``name`` — e.g. slug ``git2`` ships
    ``name: git``, and collision-suffixed slugs like ``skill-git-scm`` are common. The
    loader requires the frontmatter name to match the directory name, so a mismatched
    skill would silently fail to load after a "successful" install. Normalising the
    name here keeps the directory name (the slug, used everywhere as the identifier)
    authoritative. Only the ``name`` line is rewritten; all other frontmatter and the
    body are left untouched.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return
    raw = skill_md.read_text()

    split = _split_frontmatter(raw)
    if split is None:
        # No parseable frontmatter — prepend a minimal one.
        skill_md.write_text(f"---\nname: {name}\n---\n\n{raw.lstrip()}\n")
        return

    frontmatter, remainder = split
    name_line = re.compile(r"^name:.*(?:\n|$)", re.MULTILINE)
    if name_line.search(frontmatter):
        frontmatter = name_line.sub(f"name: {name}\n", frontmatter, count=1)
    else:
        # frontmatter already opens with the newline after '---', so it doubles as the
        # separator before the inserted name line.
        frontmatter = f"\nname: {name}{frontmatter}"

    skill_md.write_text(f"---{frontmatter}{remainder}")


def list_skill_files(skill_dir: Path) -> list[str]:
    """List supporting files relative to skill_dir, excluding SKILL.md."""
    result = []
    for f in sorted(skill_dir.rglob("*")):
        if f.is_file() and f.name != "SKILL.md":
            result.append(str(f.relative_to(skill_dir)))
    return result


def load_skill(name: str, skills_dir: Path | None = None) -> SkillDetail | None:
    """Load a single skill by name with its supporting file list.

    Returns None if the skill does not exist or fails to parse.
    """
    directory = skills_dir or SKILLS_DIR
    skill_dir = directory / name
    skill = _load_skill(skill_dir)
    if skill is None:
        return None
    files = list_skill_files(skill_dir)
    return SkillDetail(
        name=skill.name,
        description=skill.description,
        body=skill.body,
        files=files,
    )


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


def _load_skill(skill_dir: Path) -> Skill | None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        logger.warning("skills_loader skill_dir=%s missing SKILL.md — skipped", skill_dir.name)
        return None

    try:
        raw = skill_md.read_text()
        frontmatter, body = _parse_skill_md(raw)
    except Exception:
        logger.warning(
            "skills_loader skill_dir=%s parse error — skipped", skill_dir.name, exc_info=True
        )
        return None

    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if not name or not description:
        logger.warning(
            "skills_loader skill_dir=%s missing name or description — skipped", skill_dir.name
        )
        return None

    if name != skill_dir.name:
        logger.warning(
            "skills_loader skill_dir=%s name mismatch (frontmatter=%s) — skipped",
            skill_dir.name,
            name,
        )
        return None

    return Skill(name=name, description=description, body=body.strip())


def _split_frontmatter(raw: str) -> tuple[str, str] | None:
    """Split SKILL.md at its YAML '---' fences.

    Returns ``(frontmatter_text, remainder)`` where ``frontmatter_text`` is the raw
    text between the opening and closing '---' lines and ``remainder`` is the closing
    '---' line plus everything after it — so ``"---" + frontmatter_text + remainder``
    reproduces the input exactly. Returns None when there is no closing fence.

    The closing fence must sit at the start of a line, so '---' appearing inside a
    frontmatter value does not split the block prematurely.
    """
    if not raw.startswith("---"):
        return None
    m = re.search(r"^---\s*$", raw[3:], re.MULTILINE)
    if m is None:
        return None
    return raw[3 : m.start() + 3], raw[m.start() + 3 :]


def _parse_skill_md(raw: str) -> tuple[dict, str]:
    """Parse YAML frontmatter + body from SKILL.md content."""
    split = _split_frontmatter(raw)
    if split is None:
        return {}, raw

    frontmatter_text, remainder = split
    _, _, body = remainder.partition("\n")  # drop the closing '---' line
    frontmatter = yaml.safe_load(frontmatter_text.strip()) or {}
    return frontmatter, body.strip()
