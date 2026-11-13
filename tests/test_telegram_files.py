"""Files on Telegram: what comes in (media types, albums) and what goes out (by type)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from aug.api.interfaces.base import FileContent, TextContent
from aug.api.interfaces.telegram.interface import (
    TelegramInterface,
    _AlbumCollector,
    _send_file,
    _send_method,
)
from aug.core.harness import FileEmitted, FilePart, TextDelta, TurnEnded


def _message(**media) -> MagicMock:
    """A Telegram message with no media except *media*."""
    msg = MagicMock()
    for attr in ("voice", "audio", "video", "video_note", "animation", "sticker", "document"):
        setattr(msg, attr, None)
    msg.photo = []
    msg.location = None
    msg.text = None
    msg.caption = None
    msg.forward_origin = None
    msg.media_group_id = None
    msg.message_thread_id = None
    for key, value in media.items():
        setattr(msg, key, value)
    return msg


def _update(msg: MagicMock, chat_id: int = 111) -> MagicMock:
    update = MagicMock()
    update.message = msg
    update.edited_message = None
    update.effective_chat.id = chat_id
    update.effective_user.id = chat_id
    return update


def _media(data: bytes, **attrs) -> MagicMock:
    tg_file = MagicMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    obj = MagicMock()
    obj.get_file = AsyncMock(return_value=tg_file)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


@pytest.fixture()
def interface():
    tg = TelegramInterface()
    with (
        patch("aug.api.interfaces.telegram.interface.is_allowed", return_value=True),
        patch.object(TelegramInterface, "get_agent_version", return_value="v"),
    ):
        yield tg


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("media", "media_type", "filename"),
    [
        (
            {"video": _media(b"mp4", file_name="clip.mp4", mime_type="video/mp4")},
            "video/mp4",
            "clip.mp4",
        ),
        ({"video_note": _media(b"mp4")}, "video/mp4", "video_note.mp4"),
        (
            {
                "animation": _media(b"gif", file_name="fun.mp4", mime_type="video/mp4"),
                "document": _media(b"gif", file_name="fun.mp4", mime_type="video/mp4"),
            },
            "video/mp4",
            "fun.mp4",
        ),
        (
            {"audio": _media(b"mp3", file_name="song.mp3", mime_type="audio/mpeg")},
            "audio/mpeg",
            "song.mp3",
        ),
        (
            {"document": _media(b"%PDF", file_name="cv.pdf", mime_type="application/pdf")},
            "application/pdf",
            "cv.pdf",
        ),
    ],
)
async def test_each_media_type_lands_in_the_inbox(interface, media, media_type, filename):
    incoming = await interface.receive_message(_update(_message(**media)))

    [part] = incoming.parts
    assert isinstance(part, FileContent)
    assert (part.mime_type, part.filename) == (media_type, filename)
    assert "inbox" in Path(part.path).parts


async def test_a_document_without_a_type_gets_one_from_its_name(interface):
    doc = _media(b"data", file_name="IMG_7.HEIC", mime_type="application/octet-stream")

    incoming = await interface.receive_message(_update(_message(document=doc)))

    # Not decodable, so kept as sent — but typed as HEIC, which is what made it normalisable.
    assert incoming.parts[0].mime_type == "image/heic"


async def test_a_voice_note_is_marked_for_transcription(interface):
    voice = _media(b"ogg", mime_type="audio/ogg")

    incoming = await interface.receive_message(_update(_message(voice=voice)))

    assert incoming.parts[0].transcribe is True


async def test_a_caption_follows_its_file(interface):
    photo = _media(b"jpg")
    incoming = await interface.receive_message(_update(_message(photo=[photo], caption="look")))

    assert [type(p) for p in incoming.parts] == [FileContent, TextContent]
    assert incoming.parts[1].text == "look"


async def test_a_file_over_the_download_limit_becomes_a_note(interface):
    doc = _media(b"", file_name="movie.mkv", mime_type="video/x-matroska")
    doc.get_file = AsyncMock(side_effect=BadRequest("File is too big"))

    incoming = await interface.receive_message(_update(_message(document=doc)))

    [part] = incoming.parts
    assert isinstance(part, TextContent)
    assert "movie.mkv" in part.text and "20 MB" in part.text


# ---------------------------------------------------------------------------
# Albums
# ---------------------------------------------------------------------------


def _album_item(message_id: int, group: str = "g1", caption: str | None = None) -> MagicMock:
    msg = _message(photo=[_media(f"jpg{message_id}".encode())], caption=caption)
    msg.media_group_id = group
    msg.message_id = message_id
    return _update(msg)


async def test_album_items_wait_for_the_album_to_finish():
    received: list[list] = []

    async def on_album(updates):
        received.append(updates)

    albums = _AlbumCollector(on_album, wait=0.05)
    albums.add(_album_item(2), "tg-111-0")
    await asyncio.sleep(0.03)
    albums.add(_album_item(1), "tg-111-0")  # restarts the wait
    await asyncio.sleep(0.03)
    assert received == []  # 0.06s since the first item, but only 0.03s since the last
    await asyncio.sleep(0.05)

    assert [[u.message.message_id for u in album] for album in received] == [[1, 2]]


async def test_two_albums_are_kept_apart():
    received: list[list] = []

    async def on_album(updates):
        received.append([u.message.message_id for u in updates])

    albums = _AlbumCollector(on_album, wait=0.02)
    albums.add(_album_item(1, group="a"), "tg-111-0")
    albums.add(_album_item(2, group="b"), "tg-111-0")
    albums.add(_album_item(3, group="a"), "tg-111-0")
    await asyncio.sleep(0.06)

    assert sorted(received) == [[1, 3], [2]]


async def test_an_album_becomes_one_message(interface):
    handled = []
    with patch.object(
        TelegramInterface, "handle", new=AsyncMock(side_effect=lambda m, c: handled.append(m))
    ):
        await interface._handle_album(
            [_album_item(1, caption="our trip"), _album_item(2), _album_item(3)]
        )

    [incoming] = handled
    files = [p for p in incoming.parts if isinstance(p, FileContent)]
    texts = [p.text for p in incoming.parts if isinstance(p, TextContent)]
    assert [Path(f.path).read_bytes() for f in files] == [b"jpg1", b"jpg2", b"jpg3"]
    assert texts == ["our trip"]


async def test_one_failed_item_does_not_lose_the_album(interface):
    broken = _album_item(2)
    broken.message.photo[0].get_file = AsyncMock(side_effect=OSError("network down"))
    handled = []
    with patch.object(
        TelegramInterface, "handle", new=AsyncMock(side_effect=lambda m, c: handled.append(m))
    ):
        await interface._handle_album([_album_item(1, caption="trip"), broken, _album_item(3)])

    [incoming] = handled
    files = [p for p in incoming.parts if isinstance(p, FileContent)]
    notes = [p.text for p in incoming.parts if isinstance(p, TextContent)]
    assert [Path(f.path).read_bytes() for f in files] == [b"jpg1", b"jpg3"]
    assert any("could not be received" in n for n in notes)


async def test_a_message_sent_with_an_album_joins_its_turn(interface):
    """A forward's text (or "what's this?") arrives before the album is complete: it
    waits for the album, and both make one turn."""
    interface._albums = _AlbumCollector(interface._handle_album, wait=0.3)
    turns = []

    async def converse(self, incoming, message, context):
        turns.append(message)

    with (
        patch.object(TelegramInterface, "_converse", new=converse),
        patch("aug.api.interfaces.base._harness_for", return_value=None),
        patch("aug.api.interfaces.telegram.interface.get_context_id", return_value="tg-111-0"),
    ):
        text = _update(_message(text="[Forwarded from Ann] look at these"))
        text.message.message_id = 1
        await interface._handle_input(_album_item(2), MagicMock())
        await interface._handle_input(_album_item(3), MagicMock())
        await interface.run(text)  # buffered, then waits on the album
        await asyncio.sleep(0.6)

    [message] = turns
    assert sum(isinstance(p, FilePart) for p in message.parts) == 2
    assert "look at these" in message.text


async def test_an_album_item_routes_to_the_collector(interface):
    update = _album_item(1)
    with patch.object(TelegramInterface, "run", new=AsyncMock()) as run:
        interface._albums = MagicMock()
        await interface._handle_input(update, MagicMock())

    interface._albums.add.assert_called_once_with(update, "tg-111-0")
    run.assert_not_called()


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("media_type", "method"),
    [
        ("image/jpeg", "send_photo"),
        ("image/png", "send_photo"),
        ("image/gif", "send_animation"),
        ("image/svg+xml", "send_document"),
        ("audio/ogg", "send_voice"),
        ("audio/mpeg", "send_audio"),
        ("video/mp4", "send_video"),
        ("video/webm", "send_document"),
        ("application/pdf", "send_document"),
    ],
)
def test_send_method_follows_the_media_type(media_type, method):
    assert _send_method(media_type)[0] == method


def _file(tmp_path, media_type: str = "image/png", caption: str = "") -> FileEmitted:
    path = tmp_path / "out.bin"
    path.write_bytes(b"bytes")
    return FileEmitted(
        file_id="0123456789abcdef",
        path=path,
        media_type=media_type,
        filename="out.png",
        caption=caption,
    )


async def test_a_file_is_sent_with_its_name_and_caption(tmp_path):
    bot = MagicMock()
    bot.send_photo = AsyncMock()

    await _send_file(bot, 5, 9, _file(tmp_path, caption="hi"), silent=True)

    kwargs = bot.send_photo.call_args.kwargs
    assert kwargs["photo"] == b"bytes"
    assert (kwargs["chat_id"], kwargs["message_thread_id"]) == (5, 9)
    assert (kwargs["filename"], kwargs["caption"], kwargs["disable_notification"]) == (
        "out.png",
        "hi",
        True,
    )


async def test_a_refused_photo_goes_again_as_a_document(tmp_path):
    bot = MagicMock()
    bot.send_photo = AsyncMock(side_effect=BadRequest("Photo_invalid_dimensions"))
    bot.send_document = AsyncMock()

    await _send_file(bot, 5, None, _file(tmp_path), silent=False)

    assert bot.send_document.call_args.kwargs["document"] == b"bytes"


async def test_live_chat_sends_a_file_as_soon_as_it_is_made(interface, tmp_path):
    bot = MagicMock()
    bot.send_message_draft = AsyncMock()
    bot.send_photo = AsyncMock()
    msg = MagicMock()
    msg.get_bot.return_value = bot
    msg.message_thread_id = None
    msg.reply_text = AsyncMock()
    context = MagicMock()
    context.effective_message = msg
    context.effective_chat.id = 7

    async def events():
        yield FileEmitted(
            file_id="0123456789abcdef",
            path=_file(tmp_path).path,
            media_type="image/png",
            filename="out.png",
            parent_id="subagent-call",  # a subagent's file reaches the user too
        )
        assert bot.send_photo.await_count == 1  # before the turn goes on
        yield TextDelta(text="done")
        yield TurnEnded()

    with patch("aug.api.interfaces.telegram.interface._typing_loop", new=AsyncMock()):
        await interface.send_stream(events(), context)

    assert bot.send_photo.call_args.kwargs["chat_id"] == 7


async def test_a_push_sends_its_files_after_the_text(interface, tmp_path):
    order: list[str] = []
    bot = MagicMock()
    bot.send_message_draft = AsyncMock()
    bot.send_message = AsyncMock(side_effect=lambda **kw: order.append("text"))
    bot.send_document = AsyncMock(side_effect=lambda **kw: order.append("file"))
    interface._bot_app = MagicMock(bot=bot)

    async def events():
        yield FileEmitted(
            file_id="0123456789abcdef",
            path=_file(tmp_path).path,
            media_type="application/pdf",
            filename="report.pdf",
        )
        yield TextDelta(text="Here is your report.")
        yield TurnEnded()

    await interface.send_proactive_stream("tg-7-0", events())

    assert order == ["text", "file"]


async def test_a_file_that_cannot_be_sent_is_reported(interface, tmp_path):
    bot = MagicMock()
    bot.send_message_draft = AsyncMock()
    bot.send_message = AsyncMock()
    bot.send_document = AsyncMock(side_effect=BadRequest("Wrong file"))
    interface._bot_app = MagicMock(bot=bot)

    async def events():
        yield FileEmitted(
            file_id="0123456789abcdef",
            path=_file(tmp_path).path,
            media_type="application/pdf",
            filename="report.pdf",
        )
        yield TurnEnded()

    await interface.send_proactive_stream("tg-7-0", events())

    assert "report.pdf" in bot.send_message.call_args.kwargs["text"]
