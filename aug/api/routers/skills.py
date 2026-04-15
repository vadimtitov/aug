"""Skills router — CRUD for local skills and ClawHub install."""

import io
import logging
import shutil
import zipfile
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from aug.api.security import require_api_key
from aug.utils.skills import (
    SKILLS_DIR,
    list_skill_files,
    load_skill,
    load_skills,
    validate_name,
    write_skill_md,
)

logger = logging.getLogger(__name__)

CLAWHUB_DOWNLOAD_URL = "https://clawhub.ai/api/v1/download"

router = APIRouter(
    tags=["skills"],
    dependencies=[Depends(require_api_key)],
)


class SkillUpdateRequest(BaseModel):
    description: str | None = None
    body: str | None = None
    always_on: bool | None = None


class FileUpdateRequest(BaseModel):
    content: str


class InstallRequest(BaseModel):
    slug: str
    version: str | None = None


# ---------------------------------------------------------------------------
# GET /skills
# ---------------------------------------------------------------------------


@router.get("/skills")
async def list_skills_endpoint() -> list[dict]:
    """List all local skills with metadata."""
    index = load_skills(SKILLS_DIR)
    all_skills = index.always_on + index.on_demand
    result = []
    for skill in all_skills:
        skill_dir = SKILLS_DIR / skill.name
        file_count = len(list_skill_files(skill_dir))
        result.append(
            {
                "name": skill.name,
                "description": skill.description,
                "always_on": skill.always_on,
                "file_count": file_count,
            }
        )
    return result


# ---------------------------------------------------------------------------
# GET /skills/{name}
# ---------------------------------------------------------------------------


@router.get("/skills/{name}")
async def get_skill_endpoint(name: str) -> dict:
    """Return full detail for a local skill, including file list."""
    _require_valid_name(name)
    detail = load_skill(name, SKILLS_DIR)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )
    return {
        "name": detail.name,
        "description": detail.description,
        "body": detail.body,
        "always_on": detail.always_on,
        "files": detail.files,
    }


# ---------------------------------------------------------------------------
# GET /skills/{name}/file
# ---------------------------------------------------------------------------


@router.get("/skills/{name}/file", response_class=PlainTextResponse)
async def get_skill_file(name: str, path: str = Query(...)) -> PlainTextResponse:
    """Return raw content of a file within a skill directory."""
    _require_valid_name(name)
    skill_dir = SKILLS_DIR / name
    if not skill_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )
    target = _resolve_path(skill_dir, path)
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{path}' not found."
        )
    return PlainTextResponse(target.read_text())


# ---------------------------------------------------------------------------
# PUT /skills/{name}
# ---------------------------------------------------------------------------


@router.put("/skills/{name}")
async def update_skill(name: str, req: SkillUpdateRequest) -> dict:
    """Patch a local skill's description, body, and/or always_on flag."""
    _require_valid_name(name)
    detail = load_skill(name, SKILLS_DIR)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )

    description = req.description if req.description is not None else detail.description
    body = req.body if req.body is not None else detail.body
    always_on = req.always_on if req.always_on is not None else detail.always_on

    write_skill_md(SKILLS_DIR / name, name, description, body, always_on)
    return {"ok": True}


# ---------------------------------------------------------------------------
# PUT /skills/{name}/file
# ---------------------------------------------------------------------------


@router.put("/skills/{name}/file")
async def update_skill_file(name: str, req: FileUpdateRequest, path: str = Query(...)) -> dict:
    """Overwrite the content of a supporting file within a skill directory."""
    _require_valid_name(name)
    skill_dir = SKILLS_DIR / name
    if not skill_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )
    target = _resolve_path(skill_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content)
    return {"ok": True}


# ---------------------------------------------------------------------------
# DELETE /skills/{name}
# ---------------------------------------------------------------------------


@router.delete("/skills/{name}")
async def delete_skill_endpoint(name: str) -> dict:
    """Delete an entire skill directory."""
    _require_valid_name(name)
    skill_dir = SKILLS_DIR / name
    if not skill_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )
    shutil.rmtree(skill_dir)
    return {"ok": True}


# ---------------------------------------------------------------------------
# DELETE /skills/{name}/file
# ---------------------------------------------------------------------------


@router.delete("/skills/{name}/file")
async def delete_skill_file(name: str, path: str = Query(...)) -> dict:
    """Delete a supporting file within a skill directory."""
    _require_valid_name(name)
    skill_dir = SKILLS_DIR / name
    if not skill_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill '{name}' not found."
        )

    target = _resolve_path(skill_dir, path)

    if target == (skill_dir / "SKILL.md").resolve():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete SKILL.md. Delete the entire skill instead.",
        )
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{path}' not found."
        )
    target.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# POST /skills/{name}/install
# ---------------------------------------------------------------------------


@router.post("/skills/{name}/install")
async def install_skill(name: str, req: InstallRequest) -> dict:
    """Download a skill from ClawHub and extract it to the local skills directory."""
    _require_valid_name(name)
    params: dict[str, str] = {"slug": req.slug}
    if req.version:
        params["version"] = req.version
    else:
        params["tag"] = "latest"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(CLAWHUB_DOWNLOAD_URL, params=params)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "a moment")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"ClawHub rate limit hit. Try again in {retry_after}s.",
                )
            response.raise_for_status()
            zip_bytes = response.content
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.warning("clawhub_download_failed slug=%s error=%s", req.slug, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to download skill from ClawHub.",
        ) from exc

    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            # ClawHub zips have no top-level directory — files are at root.
            # _meta.json is clawhub provenance metadata, not part of the skill.
            if Path(member.filename).name == "_meta.json":
                continue
            relative = Path(member.filename)
            target = (skill_dir / relative).resolve()
            try:
                target.relative_to(skill_dir.resolve())
            except ValueError:
                logger.warning("install_skill zip traversal attempt member=%s", member.filename)
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member.filename))

    logger.info("install_skill installed slug=%s to %s", req.slug, skill_dir)
    return {"ok": True, "name": name}


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


def _require_valid_name(name: str) -> None:
    """Raise 422 if name fails the skill name spec."""
    if err := validate_name(name):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)


def _resolve_path(skill_dir: Path, path: str) -> Path:
    """Resolve path relative to skill_dir, raising 400 if it escapes."""
    try:
        target = (skill_dir / path).resolve()
        target.relative_to(skill_dir.resolve())
        return target
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path '{path}' escapes the skill directory.",
        ) from exc
