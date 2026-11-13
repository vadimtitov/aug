"""Image normalisation and the workspace file store."""

import io
import json

from PIL import Image

from aug.utils.media import MAX_IMAGE_EDGE, guess_media_type, normalise_image
from aug.utils.storage import FileStore, file_store


def _image(size: tuple[int, int], fmt: str, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 128) if mode == "RGBA" else (200, 30, 30)).save(
        out, format=fmt
    )
    return out.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


# ---------------------------------------------------------------------------
# normalise_image
# ---------------------------------------------------------------------------


def test_heic_becomes_jpeg_at_most_the_max_edge():
    data, media_type = normalise_image(_image((4032, 3024), "HEIF"), "image/heic")

    assert media_type == "image/jpeg"
    assert _size(data) == (MAX_IMAGE_EDGE, 1176)


def test_small_heic_is_still_reencoded():
    data, media_type = normalise_image(_image((800, 600), "HEIF"), "image/heic")

    assert media_type == "image/jpeg"
    assert _size(data) == (800, 600)


def test_huge_png_is_scaled_down_to_jpeg():
    data, media_type = normalise_image(_image((6000, 1500), "PNG"), "image/png")

    assert media_type == "image/jpeg"
    assert _size(data) == (MAX_IMAGE_EDGE, 392)


def test_transparent_png_is_flattened_onto_white():
    data, _ = normalise_image(_image((3000, 3000), "PNG", mode="RGBA"), "image/png")

    assert Image.open(io.BytesIO(data)).mode == "RGB"


def test_small_jpeg_is_left_untouched():
    original = _image((1200, 900), "JPEG")

    assert normalise_image(original, "image/jpeg") == (original, "image/jpeg")


def test_small_png_keeps_its_format():
    original = _image((500, 500), "PNG")

    assert normalise_image(original, "image/png") == (original, "image/png")


def test_unreadable_data_is_never_labelled_a_model_image():
    # Inlined, it would fail every later model call in the conversation.
    assert normalise_image(b"not an image", "image/jpeg") == (
        b"not an image",
        "application/octet-stream",
    )


def test_unreadable_data_keeps_a_type_no_model_is_sent():
    assert normalise_image(b"<svg/>", "image/svg+xml") == (b"<svg/>", "image/svg+xml")


def _animated_gif(size: tuple[int, int]) -> bytes:
    frames = [Image.new("RGB", size, c) for c in ((255, 0, 0), (0, 255, 0))]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:])
    return out.getvalue()


def test_a_small_animated_gif_is_left_alone():
    original = _animated_gif((400, 100))

    assert normalise_image(original, "image/gif") == (original, "image/gif")


def test_an_oversized_animated_gif_keeps_its_first_frame():
    data, media_type = normalise_image(_animated_gif((4000, 100)), "image/gif")

    assert media_type == "image/jpeg"
    assert _size(data) == (MAX_IMAGE_EDGE, 39)


def test_guess_media_type():
    assert guess_media_type("IMG_0001.HEIC") == "image/heic"
    assert guess_media_type("report.pdf") == "application/pdf"
    assert guess_media_type("no-extension") == "application/octet-stream"


# ---------------------------------------------------------------------------
# FileStore
# ---------------------------------------------------------------------------


async def test_inbound_image_is_normalised_and_renamed():
    stored = await file_store.save_inbound(_image((4032, 3024), "HEIF"), "IMG_1.HEIC", "image/heic")

    assert stored.media_type == "image/jpeg"
    assert stored.filename == "IMG_1.jpg"
    assert stored.path.parent.parent.name == "inbox"
    assert stored.path.name.endswith("-IMG_1.jpg")
    assert _size(stored.path.read_bytes())[0] == MAX_IMAGE_EDGE


async def test_inbound_non_image_is_stored_as_sent():
    stored = await file_store.save_inbound(b"%PDF-1.7", "My Report.pdf", "application/pdf")

    assert stored.path.read_bytes() == b"%PDF-1.7"
    assert stored.filename == "My Report.pdf"  # shown as sent; the path is made safe
    assert stored.path.name.endswith("-My_Report.pdf")
    assert stored.size == 8


async def test_outbound_snapshot_keeps_what_was_sent(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("first")
    stored = await file_store.snapshot_outbound(source, "notes.txt", "text/plain")
    source.write_text("changed later")

    assert stored.path.read_text() == "first"
    assert stored.path.parent.parent.name == "outbox"


async def test_get_finds_a_file_by_id_from_a_fresh_store():
    stored = await file_store.save_outbound(b"png", "a.png", "image/png")

    found = FileStore(file_store.root).get(stored.id)  # the index is on disk

    assert found == stored


async def test_get_refuses_ids_that_are_not_ids():
    assert file_store.get("../../settings") is None
    assert file_store.get("0123456789abcdef") is None


async def test_get_is_none_once_the_file_is_gone():
    stored = await file_store.save_outbound(b"x", "x.bin", "application/octet-stream")
    stored.path.unlink()

    assert file_store.get(stored.id) is None


async def test_index_record_never_holds_the_bytes():
    stored = await file_store.save_inbound(b"secret bytes", "a.txt", "text/plain")

    record = json.loads((file_store.root / "index" / f"{stored.id}.json").read_text())

    assert "secret bytes" not in json.dumps(record)
    assert record["sha256"] == stored.sha256


def test_the_bytes_decide_not_the_claimed_type():
    data, media_type = normalise_image(_image((800, 600), "HEIF"), "image/jpeg")

    assert media_type == "image/jpeg"
    assert Image.open(io.BytesIO(data)).format == "JPEG"


def test_a_passthrough_image_gets_its_real_type():
    original = _image((500, 500), "PNG")

    assert normalise_image(original, "image/jpeg") == (original, "image/png")


def test_an_image_over_the_byte_limit_is_reencoded(monkeypatch):
    monkeypatch.setattr("aug.utils.media.MAX_IMAGE_BYTES", 100)

    _, media_type = normalise_image(_image((500, 500), "PNG"), "image/png")

    assert media_type == "image/jpeg"


def test_a_decompression_bomb_is_left_unread_and_never_inlined(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # a 100x100 image is now a bomb
    original = _image((100, 100), "PNG")

    assert normalise_image(original, "image/png") == (original, "application/octet-stream")


async def test_stored_files_are_read_only():
    stored = await file_store.save_inbound(b"%PDF", "a.pdf", "application/pdf")

    assert stored.path.stat().st_mode & 0o222 == 0


async def test_a_malformed_index_record_is_no_file():
    stored = await file_store.save_outbound(b"x", "x.txt", "text/plain")
    (file_store.root / "index" / f"{stored.id}.json").write_text('{"id": "x"}')

    assert file_store.get(stored.id) is None


async def test_an_unreadable_upload_keeps_its_name():
    stored = await file_store.save_inbound(b"truncated", "photo.png", "image/png")

    assert (stored.filename, stored.media_type) == ("photo.png", "application/octet-stream")


async def test_get_never_follows_an_index_out_of_the_store(tmp_path):
    secret = tmp_path / "id_ed25519"
    secret.write_text("PRIVATE KEY")
    stored = await file_store.save_outbound(b"x", "x.txt", "text/plain")
    index = file_store.root / "index" / f"{stored.id}.json"
    record = json.loads(index.read_text())
    index.write_text(json.dumps({**record, "path": str(secret)}))

    assert file_store.get(stored.id) is None
