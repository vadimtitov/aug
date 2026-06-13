"""Skill dependencies — declaration, resolution, and warm-up.

The dependency model is the agentskills.io one: a skill's Python scripts declare the
packages they need inline, as PEP 723 metadata (a ``# /// script`` block), and run with
``uv run``. ``uv`` resolves, downloads, and builds the script's environment on first run
and caches it in ``UV_CACHE_DIR`` (a persistent volume that survives container rebuilds).

This module is the single home for that concept:

* ``inject_dependencies`` writes/replaces the PEP 723 block from a plain list of packages,
  so callers never hand-author TOML.
* ``read_pep723_deps`` / ``find_pep723_scripts`` read declared dependencies back out, for
  awareness (e.g. surfacing them when a skill is loaded).
* ``resolve_dependencies`` installs and caches one script's dependencies now, returning a
  structured result so a tool can report success or a precise failure to the agent.
* ``warm_all_skills`` / ``warm_skill_dir`` pre-resolve every installed skill's dependencies
  ahead of time (at startup and after a ClawHub install), so the first agent run after a
  rebuild never stalls on downloads.

Warm-up is best-effort: a skill whose dependencies fail to resolve is logged and skipped,
never raised — it must not block startup or break an install.
"""

import logging
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from aug.utils.skills import SKILLS_DIR

logger = logging.getLogger(__name__)

# PEP 723 inline script metadata: a block of '#'-prefixed lines opening with '# /// script'
# and closing with '# ///'.
_PEP723_OPEN_RE = re.compile(r"^# /// script", re.MULTILINE)
_PEP723_BLOCK_RE = re.compile(r"^# /// script[ \t]*\n(?:#.*\n)*?# ///[ \t]*\n?", re.MULTILINE)

# uv must resolve, download, and build the environment; heavy dependency trees can be slow.
_RESOLVE_TIMEOUT_S = 600


@dataclass(frozen=True)
class DependencyResult:
    """Outcome of resolving one script's dependencies."""

    ok: bool
    detail: str


def inject_dependencies(content: str, deps: list[str]) -> str:
    """Return ``content`` with a PEP 723 block declaring exactly ``deps`` at the top.

    Replaces an existing block if present; otherwise inserts one (after a shebang line,
    if any). Passing an empty list removes the block. Callers pass plain requirement
    strings (e.g. ``"httpx>=0.27"``) and never write the TOML themselves.
    """
    content = content.replace("\r\n", "\n")  # the block markers are matched on '\n' boundaries
    shebang, body = _split_shebang(_PEP723_BLOCK_RE.sub("", content, count=1).lstrip("\n"))
    if not deps:
        return f"{shebang}{body}"
    block = _render_pep723_block(deps)
    return f"{shebang}{block}\n{body}" if body else f"{shebang}{block}\n"


def has_pep723_block(content: str) -> bool:
    """True if ``content`` already contains a PEP 723 inline-metadata block."""
    return _PEP723_OPEN_RE.search(content.replace("\r\n", "\n")) is not None


def read_pep723_deps(script: Path) -> list[str]:
    """Return the dependencies declared in a script's PEP 723 block (empty if none)."""
    try:
        text = script.read_text(errors="ignore").replace("\r\n", "\n")
    except OSError:
        return []
    match = _PEP723_BLOCK_RE.search(text)
    if match is None:
        return []
    toml = "\n".join(
        re.sub(r"^#\s?", "", line)
        for line in match.group(0).splitlines()
        if not line.startswith("# ///")
    )
    try:
        data = tomllib.loads(toml)
    except tomllib.TOMLDecodeError:
        return []
    deps = data.get("dependencies", [])
    return [str(d) for d in deps] if isinstance(deps, list) else []


def find_pep723_scripts(root: Path) -> list[Path]:
    """Return every ``*.py`` under ``root`` that declares PEP 723 inline dependencies."""
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*.py")) if _is_pep723(path)]


def resolve_dependencies(script: Path) -> DependencyResult:
    """Install and cache one script's PEP 723 dependencies now (via ``uv sync --script``).

    ``uv sync --script`` builds the same cached environment that ``uv run <script>`` later
    reuses, so a synced script runs offline (verified: a warmed script runs with
    ``uv run --offline``).
    """
    try:
        result = subprocess.run(
            ["uv", "sync", "--script", str(script)],
            capture_output=True,
            text=True,
            timeout=_RESOLVE_TIMEOUT_S,
        )
    except FileNotFoundError:
        return DependencyResult(False, "uv is not available — cannot install dependencies.")
    except subprocess.TimeoutExpired:
        return DependencyResult(False, "Dependency installation timed out.")

    if result.returncode != 0:
        return DependencyResult(False, (result.stderr or result.stdout or "").strip()[:800])
    return DependencyResult(True, "")


def warm_all_skills() -> None:
    """Pre-resolve PEP 723 dependencies for every installed skill. Blocking, best-effort."""
    warm_skill_dir(SKILLS_DIR)


def warm_skill_dir(root: Path) -> None:
    """Pre-resolve PEP 723 dependencies for every script under ``root``. Blocking, best-effort."""
    scripts = find_pep723_scripts(root)
    if not scripts:
        return
    logger.info("skill_deps warming %d PEP 723 script(s) under %s", len(scripts), root)
    warmed = 0
    for script in scripts:
        result = resolve_dependencies(script)
        if result.ok:
            warmed += 1
            logger.info("skill_deps warmed script=%s", script)
        else:
            logger.warning("skill_deps warm failed script=%s detail=%s", script, result.detail)
    logger.info("skill_deps warm-up done — %d/%d script(s) resolved", warmed, len(scripts))


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


def _is_pep723(path: Path) -> bool:
    try:
        return _PEP723_OPEN_RE.search(path.read_text(errors="ignore")) is not None
    except OSError:
        return False


def _split_shebang(content: str) -> tuple[str, str]:
    """Split a leading ``#!`` shebang line off ``content``. Returns (shebang_or_empty, rest)."""
    if not content.startswith("#!"):
        return "", content
    shebang, _, rest = content.partition("\n")
    return f"{shebang}\n", rest.lstrip("\n")


def _render_pep723_block(deps: list[str]) -> str:
    lines = ["# /// script", "# dependencies = ["]
    lines += [f'#   "{dep}",' for dep in deps]
    lines += ["# ]", "# ///"]
    return "\n".join(lines)
