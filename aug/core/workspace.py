"""The agent's workspace: persona, memory, skills and files, shared by every agent version.

    workspace/AGENTS.md          persona and approach; edited by the user (or the agent
                                 when asked), never by the dream
    workspace/memory/            long-term memory, written only by the dream
    workspace/memory/MEMORY.md   the index: one line per memory file
    workspace/skills/<name>/     agentskills.io skills
    workspace/files/             inbox and outbox (aug.utils.storage)

``init_workspace()`` runs at startup.  It creates whatever is missing and, once, copies
the pre-workspace folders (``/app/data/memory``, ``/app/data/skills``) in.  It never
overwrites anything, so running it again changes nothing; the old folders are left
where they are.
"""

import logging
import shutil
from pathlib import Path

from aug.core.memory import cap_memory_index, init_memory_files
from aug.core.prompts import MEMORY_INDEX_DESCRIPTIONS, build_session_prompt
from aug.utils.data import AGENTS_MD, DATA_DIR, MEMORY_DIR, MEMORY_INDEX, SKILLS_DIR, WORKSPACE_DIR
from aug.utils.skills import load_skills

logger = logging.getLogger(__name__)

AGENTS_TEMPLATE = Path(__file__).parent / "workspace_template" / "AGENTS.md"

_LEGACY_MEMORY_DIR = DATA_DIR / "memory"
_LEGACY_SKILLS_DIR = DATA_DIR / "skills"
# The template's persona is everything above this heading; a migrated install keeps
# its own persona (the old self.md) in its place.
_PERSONA_END = "\n## "


def init_workspace() -> None:
    """Create the workspace, migrating the pre-workspace folders into it once."""
    for folder in (WORKSPACE_DIR, MEMORY_DIR, SKILLS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    _migrate_memory()
    _migrate_skills()
    if not AGENTS_MD.exists():
        AGENTS_MD.write_text(_initial_agents_md())
        logger.info("workspace: wrote %s", AGENTS_MD)
    init_memory_files()
    if not MEMORY_INDEX.exists():
        MEMORY_INDEX.write_text(initial_memory_index())
        logger.info("workspace: wrote %s", MEMORY_INDEX)


def session_prompt(interface: str, *, describes_images: bool = False) -> str:
    """The system prompt for a new session, from the workspace as it is now.  Built once
    per session and kept with it, so nothing written later changes a running chat."""
    return build_session_prompt(
        workspace=str(WORKSPACE_DIR),
        agents_md=_read(AGENTS_MD),
        interface=interface,
        memory_index=cap_memory_index(_read(MEMORY_INDEX)),
        skills=[(s.name, s.description) for s in load_skills(SKILLS_DIR)],
        describes_images=describes_images,
    )


def initial_memory_index() -> str:
    """A MEMORY.md for the files in memory/ now, until the dream writes its own."""
    lines = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        if path == MEMORY_INDEX:
            continue
        description = MEMORY_INDEX_DESCRIPTIONS.get(path.name)
        lines.append(f"- {path.name}: {description}" if description else f"- {path.name}")
    return "\n".join(lines) + "\n"


def _migrate_memory() -> None:
    if not _LEGACY_MEMORY_DIR.is_dir():
        return
    for source in sorted(_LEGACY_MEMORY_DIR.glob("*.md")):
        # An old memory.md (long unused) would take MEMORY.md's place on a filesystem
        # that ignores case.
        if source.name.casefold() == MEMORY_INDEX.name.casefold():
            continue
        target = MEMORY_DIR / source.name
        if not target.exists():
            shutil.copy2(source, target)
            logger.info("workspace: copied %s to %s", source, target)


def _migrate_skills() -> None:
    if not _LEGACY_SKILLS_DIR.is_dir():
        return
    for source in sorted(_LEGACY_SKILLS_DIR.iterdir()):
        target = SKILLS_DIR / source.name
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target, symlinks=True)
            logger.info("workspace: copied skill %s", source.name)


def _initial_agents_md() -> str:
    """The template, with the old self.md as the persona when there is one."""
    template = AGENTS_TEMPLATE.read_text()
    legacy_self = _LEGACY_MEMORY_DIR / "self.md"
    persona = legacy_self.read_text().strip() if legacy_self.is_file() else ""
    if not persona:
        return template
    title, _, rest = template.partition("\n\n")
    rules = rest[rest.index(_PERSONA_END) + 1 :]
    return f"{title}\n\n{persona}\n\n{rules}"


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""
