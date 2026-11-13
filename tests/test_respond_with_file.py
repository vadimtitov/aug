"""respond_with_file: the one way a file reaches the user."""

from unittest.mock import patch

import pytest

from aug.core.tools import blocked_paths
from aug.core.tools.respond_with_file import respond_with_file
from aug.utils.file_settings import AppSettings, BashToolSettings, ToolSettings


async def _send(path, caption: str = "") -> str:
    return await respond_with_file.ainvoke({"path": str(path), "caption": caption})


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A data dir whose keys, hushed store and settings.json are blocked, as on /app/data."""
    root = tmp_path / "data"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "id_ed25519").write_text("PRIVATE KEY")
    (root / ".hushed").mkdir()
    (root / ".hushed" / "key").write_text("AGE-SECRET-KEY")
    (root / "settings.json").write_text("{}")
    monkeypatch.setattr(
        blocked_paths,
        "BLOCKED_PATHS",
        (root / "keys", root / ".hushed", root / "settings.json"),
    )
    return root


async def test_sends_a_snapshot_with_caption(tmp_path, sent_files):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.7 body")

    result = await _send(report, caption="Your report")

    assert "Sent report.pdf" in result
    [file] = sent_files
    assert file.filename == "report.pdf"
    assert file.media_type == "application/pdf"
    assert file.caption == "Your report"
    assert file.path != report and file.path.read_bytes() == b"%PDF-1.7 body"
    assert "outbox" in file.path.parts


@pytest.mark.parametrize("name", ["keys/id_ed25519", ".hushed/key", "settings.json"])
async def test_refuses_blocked_paths(data_dir, sent_files, name):
    result = await _send(data_dir / name)

    assert "NOT sent" in result
    assert not sent_files


async def test_refuses_a_symlink_into_a_blocked_dir(data_dir, tmp_path, sent_files):
    link = tmp_path / "innocent.txt"
    link.symlink_to(data_dir / "keys" / "id_ed25519")

    result = await _send(link)

    assert "NOT sent" in result
    assert not sent_files


async def test_refuses_paths_matching_the_users_blacklist(tmp_path, sent_files):
    secret = tmp_path / "wallet.dat"
    secret.write_text("coins")
    settings = AppSettings(tools=ToolSettings(bash=BashToolSettings(blacklist=[r"wallet\.dat"])))

    with patch("aug.core.tools.blocked_paths.load_settings", return_value=settings):
        result = await _send(secret)

    assert "blacklist" in result
    assert not sent_files


async def test_missing_file_is_an_explicit_failure(tmp_path, sent_files):
    result = await _send(tmp_path / "nope.txt")

    assert "NOT sent" in result and "no file" in result
    assert not sent_files


async def test_empty_file_is_an_explicit_failure(tmp_path, sent_files):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")

    assert "empty" in await _send(empty)
    assert not sent_files


async def test_oversized_file_is_refused(tmp_path, sent_files):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x")

    with patch("aug.core.tools.respond_with_file.MAX_FILE_BYTES", 0):
        result = await _send(big)

    assert "limit" in result
    assert not sent_files


async def test_outside_a_turn_nothing_claims_to_be_sent(tmp_path):
    report = tmp_path / "report.txt"
    report.write_text("hi")

    assert "NOT sent" in await _send(report)
