"""Tests for GET/PUT/DELETE /skills and POST /skills/{name}/install endpoints."""

import io
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aug.api.interfaces.telegram import TelegramInterface
from aug.app import create_app

_HEADERS = {"X-API-Key": "test-api-key"}


@asynccontextmanager
async def _async_ctx(value):
    yield value


@pytest.fixture()
def client():
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    mock_checkpointer = MagicMock()

    with (
        patch("aug.app.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("aug.app._checkpointer_context", return_value=_async_ctx(mock_checkpointer)),
        patch("aug.app.init_memory_files"),
        patch("aug.app.start_consolidation_scheduler", new=AsyncMock(return_value=MagicMock())),
        patch.object(TelegramInterface, "start_polling", new=AsyncMock()),
        patch.object(TelegramInterface, "stop_polling", new=AsyncMock()),
    ):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c


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
# GET /skills
# ---------------------------------------------------------------------------


def test_list_skills_returns_skills(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body text")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills", headers=_HEADERS)
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["name"] == "my-skill"
    assert items[0]["description"] == "Does X"
    assert items[0]["always_on"] is False
    assert items[0]["file_count"] == 0


def test_list_skills_counts_supporting_files(client: TestClient, tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "my-skill", "Does X", "body")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("echo hi")
    (skill_dir / "scripts" / "helper.py").write_text("pass")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills", headers=_HEADERS)
    assert response.json()[0]["file_count"] == 2


def test_list_skills_returns_empty_when_no_skills(client: TestClient, tmp_path: Path) -> None:
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills", headers=_HEADERS)
    assert response.status_code == 200
    assert response.json() == []


def test_list_skills_requires_auth(client: TestClient) -> None:
    response = client.get("/skills")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /skills/{name}
# ---------------------------------------------------------------------------


def test_get_skill_returns_detail(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "Follow these steps.")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill", headers=_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "my-skill"
    assert data["description"] == "Does X"
    assert data["body"] == "Follow these steps."
    assert data["always_on"] is False
    assert data["files"] == []


def test_get_skill_includes_file_list(client: TestClient, tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "my-skill", "Does X", "body")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("echo hi")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill", headers=_HEADERS)
    assert "scripts/run.sh" in response.json()["files"]


def test_get_skill_returns_404_for_missing(client: TestClient, tmp_path: Path) -> None:
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/nonexistent", headers=_HEADERS)
    assert response.status_code == 404


def test_get_skill_requires_auth(client: TestClient) -> None:
    response = client.get("/skills/any")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /skills/{name}/file
# ---------------------------------------------------------------------------


def test_get_skill_file_returns_content(client: TestClient, tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "my-skill", "Does X", "body")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("echo hello")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill/file?path=scripts/run.sh", headers=_HEADERS)
    assert response.status_code == 200
    assert response.text == "echo hello"


def test_get_skill_file_returns_skill_md(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body content")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill/file?path=SKILL.md", headers=_HEADERS)
    assert response.status_code == 200
    assert "body content" in response.text


def test_get_skill_file_returns_404_for_missing(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill/file?path=missing.txt", headers=_HEADERS)
    assert response.status_code == 404


def test_get_skill_file_rejects_path_traversal(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.get("/skills/my-skill/file?path=../other/secret.txt", headers=_HEADERS)
    assert response.status_code == 400


def test_get_skill_file_requires_auth(client: TestClient) -> None:
    response = client.get("/skills/any/file?path=SKILL.md")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# PUT /skills/{name}
# ---------------------------------------------------------------------------


def test_update_skill_body(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "old body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/my-skill",
            json={"body": "new body"},
            headers=_HEADERS,
        )
    assert response.status_code == 200
    content = (tmp_path / "my-skill" / "SKILL.md").read_text()
    assert "new body" in content
    assert "old body" not in content


def test_update_skill_description(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "old description", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/my-skill",
            json={"description": "new description"},
            headers=_HEADERS,
        )
    assert response.status_code == 200
    content = (tmp_path / "my-skill" / "SKILL.md").read_text()
    assert "new description" in content
    assert "old description" not in content


def test_update_skill_always_on_toggle(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "short body", always_on=False)
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/my-skill",
            json={"always_on": True},
            headers=_HEADERS,
        )
    assert response.status_code == 200
    content = (tmp_path / "my-skill" / "SKILL.md").read_text()
    assert "always_on" in content


def test_update_skill_preserves_unpatched_fields(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "original desc", "original body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        client.put("/skills/my-skill", json={"body": "new body"}, headers=_HEADERS)
    content = (tmp_path / "my-skill" / "SKILL.md").read_text()
    assert "original desc" in content
    assert "new body" in content


def test_update_skill_returns_404_for_missing(client: TestClient, tmp_path: Path) -> None:
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put("/skills/nonexistent", json={"body": "x"}, headers=_HEADERS)
    assert response.status_code == 404


def test_update_skill_requires_auth(client: TestClient) -> None:
    response = client.put("/skills/any", json={"body": "x"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# PUT /skills/{name}/file
# ---------------------------------------------------------------------------


def test_update_skill_file_writes_content(client: TestClient, tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "my-skill", "Does X", "body")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("old content")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/my-skill/file?path=scripts/run.sh",
            json={"content": "new content"},
            headers=_HEADERS,
        )
    assert response.status_code == 200
    assert (skill_dir / "scripts" / "run.sh").read_text() == "new content"


def test_update_skill_file_rejects_path_traversal(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/my-skill/file?path=../evil.sh",
            json={"content": "evil"},
            headers=_HEADERS,
        )
    assert response.status_code == 400


def test_update_skill_file_returns_404_for_missing_skill(
    client: TestClient, tmp_path: Path
) -> None:
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.put(
            "/skills/nonexistent/file?path=scripts/run.sh",
            json={"content": "x"},
            headers=_HEADERS,
        )
    assert response.status_code == 404


def test_update_skill_file_requires_auth(client: TestClient) -> None:
    response = client.put("/skills/any/file?path=f.sh", json={"content": "x"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /skills/{name}
# ---------------------------------------------------------------------------


def test_delete_skill_removes_directory(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/my-skill", headers=_HEADERS)
    assert response.status_code == 200
    assert not (tmp_path / "my-skill").exists()


def test_delete_skill_returns_404_for_missing(client: TestClient, tmp_path: Path) -> None:
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/nonexistent", headers=_HEADERS)
    assert response.status_code == 404


def test_delete_skill_requires_auth(client: TestClient) -> None:
    response = client.delete("/skills/any")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /skills/{name}/file
# ---------------------------------------------------------------------------


def test_delete_skill_file_removes_file(client: TestClient, tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "my-skill", "Does X", "body")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("echo hi")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/my-skill/file?path=scripts/run.sh", headers=_HEADERS)
    assert response.status_code == 200
    assert not (skill_dir / "scripts" / "run.sh").exists()
    assert (skill_dir / "SKILL.md").exists()


def test_delete_skill_file_blocks_skill_md(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/my-skill/file?path=SKILL.md", headers=_HEADERS)
    assert response.status_code == 400
    assert (tmp_path / "my-skill" / "SKILL.md").exists()


def test_delete_skill_file_rejects_path_traversal(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/my-skill/file?path=../other/secret", headers=_HEADERS)
    assert response.status_code == 400


def test_delete_skill_file_returns_404_for_missing(client: TestClient, tmp_path: Path) -> None:
    _write_skill(tmp_path, "my-skill", "Does X", "body")
    with patch("aug.api.routers.skills.SKILLS_DIR", tmp_path):
        response = client.delete("/skills/my-skill/file?path=scripts/missing.sh", headers=_HEADERS)
    assert response.status_code == 404


def test_delete_skill_file_requires_auth(client: TestClient) -> None:
    response = client.delete("/skills/any/file?path=f.sh")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /skills/{name}/install
# ---------------------------------------------------------------------------


def _make_zip(skill_md: str, extra_files: dict[str, str] | None = None) -> bytes:
    """Build an in-memory zip matching the real clawhub download format (no top-level dir)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", skill_md)
        for path, content in (extra_files or {}).items():
            zf.writestr(path, content)
    return buf.getvalue()


def test_install_skill_extracts_zip(client: TestClient, tmp_path: Path) -> None:
    skill_md = "---\nname: remote-skill\ndescription: From clawhub\n---\n\nDo this.\n"
    zip_bytes = _make_zip(skill_md)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes
    mock_response.raise_for_status = MagicMock()

    with (
        patch("aug.api.routers.skills.SKILLS_DIR", tmp_path),
        patch("aug.api.routers.skills.httpx.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        response = client.post(
            "/skills/remote-skill/install",
            json={"slug": "remote-skill"},
            headers=_HEADERS,
        )

    assert response.status_code == 200
    assert (tmp_path / "remote-skill" / "SKILL.md").exists()
    assert "Do this." in (tmp_path / "remote-skill" / "SKILL.md").read_text()


def test_install_skill_extracts_supporting_files(client: TestClient, tmp_path: Path) -> None:
    skill_md = "---\nname: remote-skill\ndescription: From clawhub\n---\n\nbody\n"
    zip_bytes = _make_zip(skill_md, {"scripts/run.sh": "echo hi"})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes
    mock_response.raise_for_status = MagicMock()

    with (
        patch("aug.api.routers.skills.SKILLS_DIR", tmp_path),
        patch("aug.api.routers.skills.httpx.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        client.post(
            "/skills/remote-skill/install",
            json={"slug": "remote-skill"},
            headers=_HEADERS,
        )

    assert (tmp_path / "remote-skill" / "scripts" / "run.sh").exists()


def test_install_skill_handles_clawhub_error(client: TestClient, tmp_path: Path) -> None:
    import httpx as _httpx

    with (
        patch("aug.api.routers.skills.SKILLS_DIR", tmp_path),
        patch("aug.api.routers.skills.httpx.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=_httpx.HTTPError("timeout"))
        mock_cls.return_value = mock_client

        response = client.post(
            "/skills/remote-skill/install",
            json={"slug": "remote-skill"},
            headers=_HEADERS,
        )

    assert response.status_code == 502


def test_install_skill_requires_auth(client: TestClient) -> None:
    response = client.post("/skills/any/install", json={"slug": "any"})
    assert response.status_code == 401
