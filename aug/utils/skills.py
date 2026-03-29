"""Skills loader — reads agentskills.io-compatible skill directories from DATA_DIR/skills/.

Each skill is a directory containing a SKILL.md file with YAML frontmatter.
AUG extension: metadata.always_on=true causes full content to be injected into every
system prompt. All other skills appear only as a one-line index entry.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from aug.utils.data import DATA_DIR

logger = logging.getLogger(__name__)

SKILLS_DIR = DATA_DIR / "skills"
ALWAYS_ON_MAX_CHARS = 1000

# Name validation per agentskills.io spec
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_CONSECUTIVE_HYPHENS_RE = re.compile(r"--")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    always_on: bool = False


@dataclass
class SkillsIndex:
    always_on: list[Skill] = field(default_factory=list)
    on_demand: list[Skill] = field(default_factory=list)


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


def load_skills() -> SkillsIndex:
    """Read all skills from SKILLS_DIR and return categorised index.

    Malformed skill directories are skipped with a warning — never crash the prompt builder.
    """
    index = SkillsIndex()
    if not SKILLS_DIR.exists():
        return index

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill = _load_skill(skill_dir)
        if skill is None:
            continue
        if skill.always_on:
            index.always_on.append(skill)
        else:
            index.on_demand.append(skill)

    return index


def build_skills_prompt(index: SkillsIndex) -> str:
    """Return the skills portion of the system prompt.

    always_on skills: full SKILL.md content injected verbatim.
    on_demand skills: one-line index entry only.
    """
    parts: list[str] = []

    for skill in index.always_on:
        parts.append(f"## {skill.name}\n{skill.body}")

    if index.on_demand:
        lines = "\n".join(f"- {s.name}: {s.description}" for s in index.on_demand)
        parts.append(f"Available skills — call get_skill(name) to load full instructions:\n{lines}")

    return "\n\n".join(parts)


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

    metadata = frontmatter.get("metadata") or {}
    always_on = str(metadata.get("always_on", "false")).lower() == "true"

    return Skill(name=name, description=description, body=body.strip(), always_on=always_on)


def _parse_skill_md(raw: str) -> tuple[dict, str]:
    """Parse YAML frontmatter + body from SKILL.md content.

    Frontmatter is delimited by '---' on its own line. Handles '---' appearing
    inside frontmatter values by looking for the closing delimiter on a line boundary.
    """
    if not raw.startswith("---"):
        return {}, raw

    # Find closing '---' at the start of a line (not just anywhere in the string)
    m = re.search(r"^---\s*$", raw[3:], re.MULTILINE)
    if not m:
        return {}, raw

    frontmatter_text = raw[3 : m.start() + 3].strip()
    body = raw[m.end() + 3 :].strip()
    frontmatter = yaml.safe_load(frontmatter_text) or {}
    return frontmatter, body
