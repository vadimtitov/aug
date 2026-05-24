"""Telegram bot — polling mode.

Telegram is treated as just another frontend via TelegramInterface(BaseInterface[Update]).

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed. If TELEGRAM_BOT_TOKEN is not set the bot
is silently disabled.

Supported input types: text, voice (transcribed), audio, photos, stickers, documents, location.
"""

import asyncio
import io
import logging
import re
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import markdown as md
from langgraph.checkpoint.base import BaseCheckpointSaver
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    Update,
)
from telegram.constants import ChatAction, MessageLimit
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    TypeHandler,
    filters,
)
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt

from aug.api.interfaces.base import (
    BaseInterface,
    FileContent,
    IncomingMessage,
    LocationContent,
    TextContent,
)
from aug.api.interfaces.telegram.ssh import _SshMixin
from aug.api.interfaces.telegram.utils import escape, get_thread_id, is_allowed, restricted
from aug.config import get_settings
from aug.core.events import (
    AgentEvent,
    ChatModelStreamEvent,
    StatusEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from aug.core.memory import run_deep_consolidation, run_light_consolidation
from aug.core.prompts import build_system_prompt
from aug.core.registry import list_agents
from aug.core.state import AgentState
from aug.core.tools.approval import (
    ApprovalDecision,
    ApprovalRequest,
    list_approvals,
    revoke_approval,
)
from aug.core.tools.display import format_tool
from aug.core.tools.output import Attachment, FileAttachment, ImageAttachment, ToolOutput
from aug.utils.data import UPLOADS_DIR
from aug.utils.file_settings import TelegramChatSettings, load_settings, save_settings
from aug.utils.skills import SKILLS_DIR, load_skills
from aug.utils.state import TelegramChatState, load_state, save_state

logger = logging.getLogger(__name__)

_SPINNER = ["🌑", "🌘", "🌗", "🌖", "🌕", "🌔", "🌓", "🌒"]
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
_MAX_TOOL_ENTRIES = 6  # max top-level tool entries shown before "…" truncation
_MAX_SUBLINES = 5  # max nested sub-tool lines shown per subagent block
# In chats without draft support each update is a real editMessageText (rate-limited),
# unlike the free sendMessageDraft previews used in DMs.  Tool status refreshes slightly
# slower than the answer text since they may compete for the chat's edit budget.
_TOOL_EDIT_THROTTLE = 3.0
_ANSWER_EDIT_THROTTLE = 2.0


@dataclass
class _ToolEntry:
    run_id: str
    label: str
    args_preview: str
    is_subagent: bool = False
    sub_lines: list[str] = field(default_factory=list)
    done: bool = False
    error: bool = False


_TG_MAX_LEN = MessageLimit.MAX_TEXT_LENGTH
_SECRET_NAME, _SECRET_VALUE = range(2)
# Chats where sendMessageDraft is not supported — learned at runtime, reset on restart
_no_draft_chats: set[int] = set()


class TelegramInterface(_SshMixin, BaseInterface[Update]):
    _debounce_window = 0.1  # 100ms — collects rapid-fire forwarded messages into one run

    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        super().__init__(checkpointer)
        self._bot_app = None

    # ------------------------------------------------------------------
    # BaseInterface implementation
    # ------------------------------------------------------------------

    async def receive_message(self, context: Update) -> IncomingMessage | None:
        msg = context.message
        if not msg:
            return None
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        user_id = context.effective_user.id  # type: ignore[union-attr]
        if not is_allowed(user_id):
            return None

        thread_id = get_thread_id(chat_id, topic_id=msg.message_thread_id)
        upload_dir = UPLOADS_DIR / thread_id

        parts = []
        if msg.voice:
            tg_file = await msg.voice.get_file()
            data = bytes(await tg_file.download_as_bytearray())
            fc = FileContent(
                path=str(upload_dir / f"voice_{msg.voice.file_unique_id}.ogg"),
                mime_type=msg.voice.mime_type or "audio/ogg",
                transcribe=True,
            )
            await fc.write(data)
            parts.append(fc)
        elif msg.audio:
            tg_file = await msg.audio.get_file()
            data = bytes(await tg_file.download_as_bytearray())
            filename = _safe_filename(msg.audio.file_name or f"audio_{msg.audio.file_unique_id}")
            fc = FileContent(
                path=str(upload_dir / filename),
                mime_type=msg.audio.mime_type or "audio/mpeg",
            )
            await fc.write(data)
            parts.append(fc)
            if msg.caption:
                parts.append(TextContent(text=msg.caption))
        elif msg.photo:
            tg_file = await msg.photo[-1].get_file()
            data = bytes(await tg_file.download_as_bytearray())
            fc = FileContent(
                path=str(upload_dir / f"photo_{msg.photo[-1].file_unique_id}.jpg"),
                mime_type="image/jpeg",
            )
            await fc.write(data)
            parts.append(fc)
            if msg.caption:
                parts.append(TextContent(text=msg.caption))
        elif msg.sticker:
            tg_file = await msg.sticker.get_file()
            data = bytes(await tg_file.download_as_bytearray())
            if msg.sticker.is_video:
                mime_type, ext = "video/webm", "webm"
            elif msg.sticker.is_animated:
                mime_type, ext = "application/x-tgsticker", "tgs"
            else:
                mime_type, ext = "image/webp", "webp"
            fc = FileContent(
                path=str(upload_dir / f"sticker_{msg.sticker.file_unique_id}.{ext}"),
                mime_type=mime_type,
            )
            await fc.write(data)
            parts.append(fc)
            if msg.sticker.emoji:
                parts.append(TextContent(text=f"[sticker: {msg.sticker.emoji}]"))
        elif msg.document:
            tg_file = await msg.document.get_file()
            data = bytes(await tg_file.download_as_bytearray())
            filename = _safe_filename(
                msg.document.file_name or f"document_{msg.document.file_unique_id}"
            )
            fc = FileContent(
                path=str(upload_dir / filename),
                mime_type=msg.document.mime_type or "application/octet-stream",
            )
            await fc.write(data)
            parts.append(fc)
            if msg.caption:
                parts.append(TextContent(text=msg.caption))
        elif msg.location:
            parts.append(
                LocationContent(latitude=msg.location.latitude, longitude=msg.location.longitude)
            )
            if msg.caption:
                parts.append(TextContent(text=msg.caption))
        elif msg.text:
            parts.append(TextContent(text=msg.text))

        if not parts:
            return None

        sender = _forward_sender(msg)
        if sender:
            parts.insert(0, TextContent(text=f"[Forwarded from {sender}]"))

        return IncomingMessage(
            parts=parts,
            interface="telegram",
            sender_id=str(chat_id),
            thread_id=thread_id,
            agent_version=load_settings()
            .telegram.chats.get(str(chat_id), TelegramChatSettings())
            .agent,
        )

    async def send_stream(self, stream: AsyncIterator[AgentEvent], context: Update) -> None:
        msg = context.effective_message
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        bot = msg.get_bot()  # type: ignore[union-attr]

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_typing_loop(context, stop_typing))

        # Rolling tool-status draft state
        tool_entries: list[_ToolEntry] = []
        run_id_to_entry: dict[str, _ToolEntry] = {}
        spin_tick = 0
        tool_status_msg = None  # editMessageText fallback (chats without draft support)
        last_tool_edit = 0.0

        # Text streaming state
        accumulated_text = ""
        use_draft = chat_id not in _no_draft_chats
        last_text_draft_time = 0.0
        stream_msg = None  # non-None only in editMessageText fallback mode
        last_stream_edit = 0.0

        stream_completed_normally = False
        spinner_stop = asyncio.Event()

        async def _push_tool_draft() -> None:
            nonlocal spin_tick, use_draft, tool_status_msg, last_tool_edit
            if not tool_entries:
                return
            spin_tick += 1
            text = _render_tool_lines(tool_entries, spin_tick)
            if use_draft:
                try:
                    await bot.send_message_draft(
                        chat_id=chat_id,
                        draft_id=1,
                        text=text,
                        message_thread_id=msg.message_thread_id,  # type: ignore[union-attr]
                    )
                    return
                except Exception as e:
                    if isinstance(e, BadRequest) and "peer_invalid" in str(e).lower():
                        if chat_id not in _no_draft_chats:
                            logger.info(
                                "sendMessageDraft not supported in chat %d (peer_invalid)"
                                " — using editMessageText for tool status",
                                chat_id,
                            )
                        _no_draft_chats.add(chat_id)
                        use_draft = False
                        # fall through to the editMessageText fallback below
                    else:
                        logger.debug("tool draft push failed", exc_info=True)
                        return
            # No-draft fallback: maintain one tool-status message, edited in place.
            now = asyncio.get_running_loop().time()
            if tool_status_msg is not None and now - last_tool_edit < _TOOL_EDIT_THROTTLE:
                return  # throttle edits to avoid flood limits
            try:
                if tool_status_msg is None:
                    tool_status_msg = await _reply_text_with_retry(
                        msg,  # type: ignore[arg-type]
                        text,
                        link_preview_options=_NO_PREVIEW,
                        disable_notification=True,
                    )
                else:
                    await tool_status_msg.edit_text(text, link_preview_options=_NO_PREVIEW)
                last_tool_edit = now
            except RetryAfter as e:
                last_tool_edit = now + e.retry_after
            except Exception:
                logger.debug("tool status edit failed", exc_info=True)

        async def _finalize_tool_draft() -> None:
            nonlocal tool_entries, run_id_to_entry, tool_status_msg, last_tool_edit
            if not tool_entries:
                return
            text = _render_tool_lines(tool_entries, spin_tick)
            tool_entries = []
            run_id_to_entry = {}
            if tool_status_msg is not None:
                # No-draft mode: freeze the live tool-status message at its final state.
                try:
                    await tool_status_msg.edit_text(text, link_preview_options=_NO_PREVIEW)
                except Exception:
                    logger.debug("tool status finalize edit failed", exc_info=True)
                tool_status_msg = None
                last_tool_edit = 0.0
                return
            try:
                await _reply_text_with_retry(
                    msg,  # type: ignore[arg-type]
                    text,
                    link_preview_options=_NO_PREVIEW,
                    disable_notification=True,
                )
            except Exception:
                logger.debug("tool draft finalize failed", exc_info=True)

        async def _spinner_loop() -> None:
            while not spinner_stop.is_set():
                # Animate every second for free drafts; in no-draft chats each tick
                # is a real editMessageText, so refresh slower to avoid flood limits.
                await asyncio.sleep(1.0 if use_draft else _TOOL_EDIT_THROTTLE)
                if spinner_stop.is_set():
                    break
                if any(not e.done for e in tool_entries):
                    await _push_tool_draft()

        spinner_task = asyncio.create_task(_spinner_loop())

        try:
            async for event in stream:
                match event:
                    case ChatModelStreamEvent(delta=delta) if delta:
                        if tool_entries:
                            await _finalize_tool_draft()

                        accumulated_text += delta
                        now = asyncio.get_running_loop().time()
                        if use_draft:
                            if now - last_text_draft_time >= 0.3:
                                try:
                                    await bot.send_message_draft(
                                        chat_id=chat_id,
                                        draft_id=1,
                                        text=_draft_preview(accumulated_text),
                                        message_thread_id=msg.message_thread_id,  # type: ignore[union-attr]
                                    )
                                    last_text_draft_time = now
                                except Exception as e:
                                    if (
                                        isinstance(e, BadRequest)
                                        and "peer_invalid" in str(e).lower()
                                    ):
                                        logger.info(
                                            "sendMessageDraft not supported in chat %d"
                                            " (peer_invalid) — falling back to editMessageText",
                                            chat_id,
                                        )
                                        _no_draft_chats.add(chat_id)
                                    else:
                                        logger.warning(
                                            "sendMessageDraft failed,"
                                            " falling back to editMessageText",
                                            exc_info=True,
                                        )
                                    use_draft = False
                                    stream_msg = await _reply_text_with_retry(
                                        msg,  # type: ignore[arg-type]
                                        _draft_preview(accumulated_text),
                                        link_preview_options=_NO_PREVIEW,
                                        disable_notification=True,
                                    )
                                    last_stream_edit = asyncio.get_running_loop().time()
                        else:
                            # No-draft chats: stream the answer via editMessageText, but at
                            # the same slow cadence as the tool status (group rate limit is
                            # ~1 msg/3s).  On RetryAfter we back off, so a truly flooded chat
                            # naturally falls back to showing the full answer at the end.
                            if stream_msg is None:
                                stream_msg = await _reply_text_with_retry(
                                    msg,  # type: ignore[arg-type]
                                    _draft_preview(accumulated_text),
                                    link_preview_options=_NO_PREVIEW,
                                    disable_notification=True,
                                )
                                last_stream_edit = now
                            elif now - last_stream_edit >= _ANSWER_EDIT_THROTTLE:
                                try:
                                    await stream_msg.edit_text(
                                        _draft_preview(accumulated_text),
                                        link_preview_options=_NO_PREVIEW,
                                    )
                                    last_stream_edit = now
                                except RetryAfter as e:
                                    last_stream_edit = now + e.retry_after
                                except Exception:
                                    logger.debug("stream edit failed", exc_info=True)

                    case ToolStartEvent(run_id=run_id, tool_name=tool_name, args=args):
                        if accumulated_text:
                            if stream_msg is not None:
                                try:
                                    await stream_msg.edit_text(
                                        _to_html(accumulated_text),
                                        parse_mode="HTML",
                                        link_preview_options=_NO_PREVIEW,
                                    )
                                except Exception:
                                    logger.debug(
                                        "stream msg edit on tool start failed", exc_info=True
                                    )
                            else:
                                try:
                                    await _reply_text_with_retry(
                                        msg,  # type: ignore[arg-type]
                                        _to_html(accumulated_text),
                                        parse_mode="HTML",
                                        link_preview_options=_NO_PREVIEW,
                                        disable_notification=True,
                                    )
                                except Exception:
                                    logger.debug("pre-tool commit failed", exc_info=True)
                            accumulated_text = ""
                            stream_msg = None

                        label, args_preview = format_tool(tool_name, args)
                        entry = _ToolEntry(
                            run_id=run_id,
                            label=label,
                            args_preview=args_preview,
                            is_subagent=(tool_name == "run_subagent"),
                        )
                        tool_entries.append(entry)
                        run_id_to_entry[run_id] = entry
                        await _push_tool_draft()

                    case ToolProgressEvent(
                        run_id=sub_run_id, tool_name=sub_tool, step=step, args=sub_args
                    ):
                        # A progress event's run_id equals the dispatching tool's own
                        # run_id, so it maps directly to that tool's entry — nesting both
                        # subagent sub-tool calls (tool_name) and browser-style step text
                        # under the right parent line.
                        parent_entry = run_id_to_entry.get(sub_run_id) if sub_run_id else None
                        if parent_entry and sub_tool:
                            sub_label, sub_preview = format_tool(sub_tool, sub_args or {})
                            inner = f"({sub_preview})" if sub_preview else "()"
                            parent_entry.sub_lines.append(f"{sub_label}{inner}")
                            await _push_tool_draft()
                        elif parent_entry and step:
                            line = step.split("\n", 1)[0].strip()
                            if len(line) > 60:
                                line = line[:60] + "…"
                            parent_entry.sub_lines.append(line)
                            await _push_tool_draft()

                    case ToolEndEvent(run_id=run_id, output=output, error=error):
                        entry = run_id_to_entry.get(run_id)
                        if entry:
                            entry.done = True
                            entry.error = error
                            await _push_tool_draft()
                        if isinstance(output, ToolOutput) and output.attachments:
                            for attachment in output.attachments:
                                await _send_attachment(msg, attachment)  # type: ignore[arg-type]

                    case StatusEvent(text=text) if text:
                        try:
                            await msg.reply_text(  # type: ignore[union-attr]
                                text,
                                disable_notification=True,
                                do_quote=False,
                            )
                        except Exception:
                            logger.debug("status event send failed", exc_info=True)

            if tool_entries:
                await _finalize_tool_draft()

            if accumulated_text:
                chunks = _chunk(accumulated_text)
                for i, chunk in enumerate(chunks):
                    html = _to_html(chunk)
                    is_last = i == len(chunks) - 1
                    if i == 0 and stream_msg is not None:
                        try:
                            await stream_msg.edit_text(  # type: ignore[union-attr]
                                html,
                                parse_mode="HTML",
                                link_preview_options=_NO_PREVIEW,
                            )
                            stream_msg = None
                            continue
                        except RetryAfter as e:
                            logger.warning(
                                "Final edit hit flood control, retrying after %ss", e.retry_after
                            )
                            await asyncio.sleep(e.retry_after)
                            try:
                                await stream_msg.edit_text(  # type: ignore[union-attr]
                                    html,
                                    parse_mode="HTML",
                                    link_preview_options=_NO_PREVIEW,
                                )
                                stream_msg = None
                                continue
                            except Exception:
                                logger.debug(
                                    "Final edit retry failed, falling back to new message",
                                    exc_info=True,
                                )
                        except BadRequest:
                            logger.warning(
                                "Final edit failed, falling back to plain", exc_info=True
                            )
                            try:
                                await stream_msg.edit_text(chunk, link_preview_options=_NO_PREVIEW)  # type: ignore[union-attr]
                                stream_msg = None
                                continue
                            except Exception:
                                logger.debug("Final plain edit also failed", exc_info=True)
                        stream_msg = None
                    try:
                        await _reply_text_with_retry(
                            msg,  # type: ignore[arg-type]
                            html,
                            parse_mode="HTML",
                            link_preview_options=_NO_PREVIEW,
                            disable_notification=not is_last,
                        )
                    except BadRequest:
                        logger.warning("Failed to send HTML, falling back to plain", exc_info=True)
                        await _reply_text_with_retry(
                            msg,  # type: ignore[arg-type]
                            chunk,
                            link_preview_options=_NO_PREVIEW,
                            disable_notification=not is_last,
                        )

            stream_completed_normally = True

        finally:
            stop_typing.set()
            typing_task.cancel()
            spinner_stop.set()
            spinner_task.cancel()
            if not stream_completed_normally and tool_entries:
                for entry in tool_entries:
                    if not entry.done:
                        entry.error = True
                        entry.done = True
                try:
                    await _finalize_tool_draft()
                except Exception:
                    logger.debug("error tool draft finalize failed", exc_info=True)

    async def send_message(self, message: str, context: Update) -> None:
        for chunk in _chunk(message):
            await _reply_text_with_retry(context.effective_message, chunk)  # type: ignore[arg-type]

    async def resolve_thread(
        self,
        thread_id: str,
        *,
        topic_name: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """Resolve a logical thread ID to a concrete Telegram thread ID.

        ``"default"`` resolves to the current DM session for *chat_id* (if
        provided) or for the first known positive chat_id in settings.
        ``"new"`` creates a Telegram forum topic in *chat_id* and returns its
        thread ID.  Any other value is returned unchanged.
        """
        if thread_id == "default" or thread_id.startswith("default:"):
            if thread_id.startswith("default:"):
                cid = int(thread_id.split(":", 1)[1])
                return get_thread_id(cid, None)
            if chat_id is not None:
                return get_thread_id(chat_id, None)
            # Legacy fallback: tasks stored before the "default:{chat_id}" encoding.
            settings = load_settings()
            for cid_str in settings.telegram.chats:
                cid = int(cid_str)
                if cid > 0:
                    logger.warning(
                        "resolve_thread: using first known chat_id %d as 'default' — "
                        "tasks should store 'default:{chat_id}' for deterministic routing",
                        cid,
                    )
                    return get_thread_id(cid, None)
            raise ValueError("No default Telegram DM chat found in settings")

        if thread_id == "new":
            if chat_id is None:
                raise ValueError("chat_id is required when thread_id='new'")
            if not self._bot_app:
                raise RuntimeError("Telegram bot is not running")
            topic = await self._bot_app.bot.create_forum_topic(
                chat_id=chat_id,
                name=topic_name or "New Topic",
            )
            return get_thread_id(chat_id, topic_id=topic.message_thread_id)

        return thread_id

    async def send_proactive(self, thread_id: str, text: str) -> None:
        """Send *text* to *thread_id* without an agent turn.

        Used for ``type="forward"`` external pushes that relay a message
        verbatim (e.g. a Home Assistant notification).
        """
        if not self._bot_app:
            raise RuntimeError("Telegram bot is not running")
        chat_id, topic_id = _parse_thread_id(thread_id)
        for chunk in _chunk(text):
            await self._bot_app.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=topic_id,
            )

    async def send_proactive_stream(
        self, thread_id: str, stream: AsyncIterator[AgentEvent]
    ) -> None:
        """Stream the agent response to *thread_id* using draft updates, then send final message.

        Uses sendMessageDraft to show tokens as they arrive (same as interactive chat).
        Falls back to send-then-edit if drafts are unsupported. Tool call messages are
        suppressed; only the final text and attachments are delivered.
        Used for ``type="agent"`` pushes and scheduled tasks.
        """
        if not self._bot_app:
            raise RuntimeError("Telegram bot is not running")
        chat_id, topic_id = _parse_thread_id(thread_id)
        bot = self._bot_app.bot

        accumulated_text = ""
        attachments: list[Attachment] = []
        use_draft = chat_id not in _no_draft_chats
        last_draft_time = 0.0
        stream_msg = None
        last_stream_edit = 0.0

        async for event in stream:
            match event:
                case ChatModelStreamEvent(delta=delta) if delta:
                    accumulated_text += delta
                    now = asyncio.get_running_loop().time()
                    if use_draft:
                        if now - last_draft_time >= 0.3:
                            try:
                                await bot.send_message_draft(
                                    chat_id=chat_id,
                                    draft_id=1,
                                    text=_draft_preview(accumulated_text),
                                    message_thread_id=topic_id,
                                )
                                last_draft_time = now
                            except Exception as e:
                                if isinstance(e, BadRequest) and "peer_invalid" in str(e).lower():
                                    _no_draft_chats.add(chat_id)
                                else:
                                    logger.warning(
                                        "proactive sendMessageDraft failed chat_id=%d,"
                                        " falling back to editMessageText",
                                        chat_id,
                                        exc_info=True,
                                    )
                                use_draft = False
                                stream_msg = await bot.send_message(
                                    chat_id=chat_id,
                                    text=_draft_preview(accumulated_text),
                                    message_thread_id=topic_id,
                                    disable_notification=True,
                                    link_preview_options=_NO_PREVIEW,
                                )
                                last_stream_edit = asyncio.get_running_loop().time()
                    else:
                        if stream_msg is None:
                            stream_msg = await bot.send_message(
                                chat_id=chat_id,
                                text=_draft_preview(accumulated_text),
                                message_thread_id=topic_id,
                                disable_notification=True,
                                link_preview_options=_NO_PREVIEW,
                            )
                            last_stream_edit = now
                        elif now - last_stream_edit >= _ANSWER_EDIT_THROTTLE:
                            try:
                                await stream_msg.edit_text(
                                    _draft_preview(accumulated_text),
                                    link_preview_options=_NO_PREVIEW,
                                )
                                last_stream_edit = now
                            except RetryAfter as e:
                                last_stream_edit = now + e.retry_after
                            except Exception:
                                logger.debug("proactive stream edit failed", exc_info=True)
                case ToolEndEvent(output=output) if (
                    isinstance(output, ToolOutput) and output.attachments
                ):
                    attachments.extend(output.attachments)

        if accumulated_text:
            for i, chunk in enumerate(_chunk(accumulated_text)):
                html = _to_html(chunk)
                if i == 0 and stream_msg is not None:
                    try:
                        await stream_msg.edit_text(
                            html, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                        )
                        stream_msg = None
                        continue
                    except Exception:
                        logger.debug("proactive final edit failed, sending new", exc_info=True)
                    stream_msg = None
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=html,
                        parse_mode="HTML",
                        message_thread_id=topic_id,
                        link_preview_options=_NO_PREVIEW,
                    )
                except Exception:
                    logger.warning(
                        "send_proactive_stream HTML failed chat_id=%d, retrying plain",
                        chat_id,
                        exc_info=True,
                    )
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            message_thread_id=topic_id,
                        )
                    except Exception:
                        logger.exception(
                            "send_proactive_stream plain fallback failed chat_id=%d, chunk skipped",
                            chat_id,
                        )

        for attachment in attachments:
            data = io.BytesIO(attachment.data)
            caption = attachment.caption or None
            if isinstance(attachment, ImageAttachment):
                await self._bot_app.bot.send_photo(
                    chat_id=chat_id,
                    photo=data,
                    caption=caption,
                    message_thread_id=topic_id,
                )
            elif isinstance(attachment, FileAttachment):
                data.name = attachment.filename
                await self._bot_app.bot.send_document(
                    chat_id=chat_id,
                    document=data,
                    caption=caption,
                    filename=attachment.filename,
                    message_thread_id=topic_id,
                )

    async def request_approval(self, request: ApprovalRequest, context: Update) -> None:
        """Send an approval prompt with inline buttons to the user."""
        msg = context.effective_message  # type: ignore[union-attr]
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        thread_id = get_thread_id(chat_id, topic_id=msg.message_thread_id)  # type: ignore[union-attr]
        agent_version = (
            load_settings().telegram.chats.get(str(chat_id), TelegramChatSettings()).agent
        )
        buttons = [
            [
                InlineKeyboardButton(
                    "✅ Run Once",
                    callback_data=f"approval:{thread_id}:{ApprovalDecision.APPROVED_ONCE.value}:{agent_version}",
                )
            ],
            [
                InlineKeyboardButton(
                    "✅✅ Allow Always",
                    callback_data=f"approval:{thread_id}:{ApprovalDecision.APPROVED_ALWAYS.value}:{agent_version}",
                )
            ],
            [
                InlineKeyboardButton(
                    "❌ Deny",
                    callback_data=f"approval:{thread_id}:{ApprovalDecision.DENIED.value}:{agent_version}",
                )
            ],
        ]
        resource_line = (
            f"<b>Target:</b> <code>{escape(request.resource)}</code>\n" if request.resource else ""
        )
        text = (
            f"⚠️ <b>Approval required</b>\n\n"
            f"<b>Tool:</b> <code>{escape(request.tool_name)}</code>\n"
            f"{resource_line}"
            f"<b>Operation:</b>\n<pre>{escape(request.operation)}</pre>"
        )
        try:
            await msg.reply_text(  # type: ignore[union-attr]
                text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(buttons),
                link_preview_options=_NO_PREVIEW,
                do_quote=False,
            )
        except Exception:
            logger.warning("approval_prompt_html_failed, retrying plain", exc_info=True)
            await msg.reply_text(  # type: ignore[union-attr]
                f"⚠️ Approval required\n{request.description}",
                reply_markup=InlineKeyboardMarkup(buttons),
                do_quote=False,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_bot(self) -> Application:
        bot_app = (  # type: ignore[arg-type]
            Application.builder()
            .token(get_settings().TELEGRAM_BOT_TOKEN)
            .concurrent_updates(True)
            .build()
        )
        bot_app.add_handler(CommandHandler("clear", self._handle_clear))
        bot_app.add_handler(CommandHandler("stop", self._handle_stop))
        bot_app.add_handler(CommandHandler("thread_id", self._handle_thread_id))
        bot_app.add_handler(
            ConversationHandler(
                entry_points=[CommandHandler("secret", self._secret_start)],
                states={
                    _SECRET_NAME: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._secret_got_name)
                    ],
                    _SECRET_VALUE: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._secret_got_value)
                    ],
                    ConversationHandler.TIMEOUT: [TypeHandler(object, self._secret_timeout)],
                },
                fallbacks=[CommandHandler("cancel", self._secret_cancel)],
                conversation_timeout=300,
            )
        )
        bot_app.add_handler(CommandHandler("version", self._handle_version))
        bot_app.add_handler(CommandHandler("skills", self._handle_skills))
        bot_app.add_handler(CommandHandler("prompt", self._handle_prompt))
        bot_app.add_handler(CommandHandler("compact", self._handle_compact))
        bot_app.add_handler(CommandHandler("consolidate", self._handle_consolidate))
        bot_app.add_handler(CommandHandler("consolidate_deep", self._handle_consolidate_deep))
        bot_app.add_handler(
            CallbackQueryHandler(self._handle_version_callback, pattern=r"^version:")
        )
        bot_app.add_handler(CallbackQueryHandler(self._handle_skills_callback, pattern=r"^skill:"))
        bot_app.add_handler(
            CallbackQueryHandler(self._handle_approval_callback, pattern=r"^approval:")
        )
        bot_app.add_handler(
            CallbackQueryHandler(
                self._handle_approvals_revoke_callback, pattern=r"^approval_revoke:"
            )
        )
        bot_app.add_handler(CommandHandler("approvals", self._handle_approvals))
        self.build_ssh_handlers(bot_app)
        bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        bot_app.add_handler(MessageHandler(filters.VOICE, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.AUDIO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.PHOTO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.Document.ALL, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.LOCATION, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.Sticker.ALL, self._handle_input))
        return bot_app

    async def start_polling(self, app) -> None:
        """Start the bot and begin polling. Called from FastAPI lifespan."""
        if not get_settings().TELEGRAM_BOT_TOKEN:
            logger.info("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
            return
        self._bot_app = self.build_bot()
        app.state.interfaces["telegram"] = self
        await self._bot_app.initialize()
        await self._bot_app.bot.set_my_commands(
            [
                ("clear", "Start a new conversation"),
                ("stop", "Stop the current run"),
                ("compact", "Summarise conversation history to free up context"),
                ("version", "Switch agent version"),
                ("skills", "Inspect skill files"),
                ("secret", "Store a secret"),
                ("prompt", "Export current system prompt as a file"),
                ("consolidate", "Run memory consolidation now"),
                ("consolidate_deep", "Run deep (weekly) memory consolidation now"),
                ("approvals", "List and revoke saved SSH command approvals"),
                ("ssh", "Manage SSH targets"),
                ("thread_id", "Show the thread ID of this chat"),
            ]
        )
        await self._bot_app.start()
        await self._bot_app.updater.start_polling()
        logger.info("Telegram bot started (polling).")

    async def stop_polling(self, app) -> None:
        """Gracefully stop the bot. Called from FastAPI lifespan shutdown."""
        if not self._bot_app:
            return
        await self._bot_app.updater.stop()
        await self._bot_app.stop()
        await self._bot_app.shutdown()
        logger.info("Telegram bot stopped.")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        await self.run(update)

    async def _handle_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await self.run(update)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    @restricted
    async def _handle_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        stopped = self.stop_run(get_thread_id(chat_id, update.effective_message.message_thread_id))  # type: ignore[union-attr]
        msg = "Stopping..." if stopped else "Nothing is running."
        await update.message.reply_text(msg)  # type: ignore[union-attr]

    @restricted
    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        topic_id = update.message.message_thread_id  # type: ignore[union-attr]
        if topic_id is not None:
            await update.message.reply_text(  # type: ignore[union-attr]
                "Each topic is its own thread. Create a new topic to start a fresh conversation."
            )
            return
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        st = load_state()
        current = st.telegram.chats.get(str(chat_id), TelegramChatState()).session
        st.telegram.chats[str(chat_id)] = TelegramChatState(session=current + 1)
        save_state(st)
        await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]

    @restricted
    async def _handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        current = load_settings().telegram.chats.get(str(chat_id), TelegramChatSettings()).agent
        agents = [a for a in list_agents() if a != "fake"]
        buttons = [
            [
                InlineKeyboardButton(
                    f"{'✅ ' if a == current else ''}{a}", callback_data=f"version:{a}"
                )
            ]
            for a in agents
        ]
        await update.message.reply_text(  # type: ignore[union-attr]
            f"Current version: <code>{escape(current)}</code>\nChoose a version:",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    @restricted
    async def _handle_version_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        agent_name = query.data.split(":", 1)[1]  # type: ignore[union-attr]
        if agent_name not in list_agents():
            await query.edit_message_text("Unknown version.")
            return
        s = load_settings()
        if str(chat_id) not in s.telegram.chats:
            s.telegram.chats[str(chat_id)] = TelegramChatSettings()
        s.telegram.chats[str(chat_id)].agent = agent_name
        save_settings(s)
        await query.edit_message_text(
            f"Switched to <code>{escape(agent_name)}</code>.", parse_mode="HTML"
        )

    @restricted
    async def _handle_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        index = await asyncio.to_thread(load_skills)
        all_skills = index.always_on + index.on_demand
        if not all_skills:
            await update.message.reply_text("No skills found.")  # type: ignore[union-attr]
            return
        buttons = [
            [InlineKeyboardButton(s.name, callback_data=f"skill:{s.name}")] for s in all_skills
        ]
        await update.message.reply_text(  # type: ignore[union-attr]
            "Choose a skill to inspect:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    @restricted
    async def _handle_skills_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        skill_name = query.data.split(":", 1)[1]  # type: ignore[union-attr]
        skill_dir = SKILLS_DIR / skill_name
        if not skill_dir.exists():
            await query.edit_message_text(f"Skill '{skill_name}' not found.")
            return
        files = sorted(f for f in skill_dir.rglob("*") if f.is_file())
        if not files:
            await query.edit_message_text(f"Skill '{skill_name}' has no files.")
            return
        await query.edit_message_text(
            f"Sending {len(files)} file(s) for skill <code>{escape(skill_name)}</code>:",
            parse_mode="HTML",
        )
        msg = update.effective_message
        for f in files:
            file_bytes = await asyncio.to_thread(f.read_bytes)
            rel = f.relative_to(skill_dir)
            await msg.reply_document(  # type: ignore[union-attr]
                document=io.BytesIO(file_bytes), filename=str(rel)
            )

    @restricted
    async def _handle_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        state = AgentState(
            messages=[],
            thread_id=get_thread_id(chat_id, update.effective_message.message_thread_id),  # type: ignore[union-attr]
            interface="telegram",
        )
        prompt_text = build_system_prompt(state)
        file = io.BytesIO(prompt_text.encode())
        file.name = "system_prompt.txt"
        await update.message.reply_document(document=file, filename="system_prompt.txt")  # type: ignore[union-attr]

    @restricted
    async def _handle_thread_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.message.message_thread_id  # type: ignore[union-attr]
        thread_id = get_thread_id(chat_id, topic_id)
        await update.message.reply_text(  # type: ignore[union-attr]
            f"<code>{escape(thread_id)}</code>",
            parse_mode="HTML",
        )

    @restricted
    async def _handle_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        agent_version = (
            load_settings().telegram.chats.get(str(chat_id), TelegramChatSettings()).agent
        )
        await update.message.reply_text("🗜 Compacting conversation…")  # type: ignore[union-attr]
        try:
            topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
            ran = await self._execute_compact(get_thread_id(chat_id, topic_id), agent_version)
            await update.message.reply_text("✅ Done." if ran else "Nothing to compact.")  # type: ignore[union-attr]
        except ValueError as e:
            await update.message.reply_text(str(e))  # type: ignore[union-attr]
        except Exception:
            logger.exception("Manual compaction failed")
            await update.message.reply_text("Compaction failed — check logs.")  # type: ignore[union-attr]

    @restricted
    async def _handle_consolidate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Running memory consolidation...")  # type: ignore[union-attr]
        try:
            ran = await run_light_consolidation()
            msg = "Done." if ran else "Nothing to consolidate — no notes."
            await update.message.reply_text(msg)  # type: ignore[union-attr]
        except Exception:
            logger.exception("Manual consolidation failed")
            await update.message.reply_text("Consolidation failed — check logs.")  # type: ignore[union-attr]

    @restricted
    async def _handle_consolidate_deep(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text("Running deep consolidation...")  # type: ignore[union-attr]
        try:
            await run_deep_consolidation()
            await update.message.reply_text("Done.")  # type: ignore[union-attr]
        except Exception:
            logger.exception("Manual deep consolidation failed")
            await update.message.reply_text("Deep consolidation failed — check logs.")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Approval handlers (Slices 4 & 5)
    # ------------------------------------------------------------------

    @restricted
    async def _handle_approval_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle user tapping Run Once / Allow Always / Deny on an approval prompt."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        # callback_data: "approval:{thread_id}:{decision}:{agent_version}"
        parts = (query.data or "").split(":", 3)
        if len(parts) != 4:
            return
        _, thread_id, decision_str, agent_version = parts

        if not thread_id.startswith(f"tg-{chat_id}-"):
            await query.edit_message_text("❌ Unauthorized.")
            return

        try:
            decision = ApprovalDecision(decision_str)
        except ValueError:
            return

        if await self.get_pending_approval(thread_id, agent_version) is None:
            await query.edit_message_text("⚠️ This approval has already been resolved.")
            return

        if decision == ApprovalDecision.DENIED:
            await query.edit_message_text("❌ Command denied.")
        elif decision == ApprovalDecision.APPROVED_ALWAYS:
            await query.edit_message_text("✅ Approved — rule saved.")
        else:
            await query.edit_message_text("✅ Approved for this run.")

        await self._execute_resume(
            thread_id=thread_id,
            agent_version=agent_version,
            sender_id=str(chat_id),
            interface="telegram",
            decision=decision,
            context=update,
        )

    @restricted
    async def _handle_approvals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List saved approval rules with per-rule Revoke buttons."""
        rules = list_approvals()
        if not rules:
            await update.message.reply_text("No saved approval rules.")  # type: ignore[union-attr]
            return
        lines = []
        buttons = []
        for i, rule in enumerate(rules, start=1):
            tool = rule.tool
            target = rule.target
            pattern = rule.pattern
            lines.append(
                f"{i}. <code>{escape(tool)}</code> @ <code>{escape(target)}</code>"
                f": <code>{escape(pattern)}</code>"
            )
            buttons.append(
                [InlineKeyboardButton(f"Revoke #{i}", callback_data=f"approval_revoke:{i - 1}")]
            )
        text = "Saved approval rules:\n\n" + "\n".join(lines)
        await update.message.reply_text(  # type: ignore[union-attr]
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
            link_preview_options=_NO_PREVIEW,
        )

    @restricted
    async def _handle_approvals_revoke_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle user tapping Revoke on an approval rule."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        parts = (query.data or "").split(":", 1)
        if len(parts) != 2:
            return
        try:
            index = int(parts[1])
            revoke_approval(index)
            await query.edit_message_text(f"Rule #{index + 1} revoked.")
        except (ValueError, IndexError):
            await query.edit_message_text(
                "Could not revoke rule — it may have already been removed."
            )

    @restricted
    async def _secret_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        msg = await update.message.reply_text("Enter secret name:")  # type: ignore[union-attr]
        context.user_data["secret_msgs"] = [update.message.message_id, msg.message_id]  # type: ignore[union-attr]
        return _SECRET_NAME

    async def _secret_got_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["secret_name"] = update.message.text  # type: ignore[union-attr]
        context.user_data["secret_msgs"].append(update.message.message_id)  # type: ignore[union-attr]
        msg = await update.message.reply_text("Enter secret value:")  # type: ignore[union-attr]
        context.user_data["secret_msgs"].append(msg.message_id)
        return _SECRET_VALUE

    async def _secret_got_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["secret_msgs"].append(update.message.message_id)  # type: ignore[union-attr]
        name = context.user_data["secret_name"]
        value = update.message.text  # type: ignore[union-attr]
        result = await asyncio.to_thread(
            subprocess.run, ["hushed", "add", name, value], capture_output=True, text=True
        )
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        for msg_id in context.user_data.pop("secret_msgs", []):
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception:
                pass
        if result.returncode == 0:
            await update.effective_chat.send_message(f"Secret {name} stored.")  # type: ignore[union-attr]
        else:
            logger.error("hushed add failed: %s", result.stderr)
            await update.effective_chat.send_message(f"Failed to store secret {name}.")  # type: ignore[union-attr]
        return ConversationHandler.END

    async def _secret_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.pop("secret_name", None)
        context.user_data.pop("secret_msgs", None)
        await update.message.reply_text("Cancelled.")  # type: ignore[union-attr]
        return ConversationHandler.END

    async def _secret_timeout(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.pop("secret_name", None)
        context.user_data.pop("secret_msgs", None)
        return ConversationHandler.END


# ---------------------------------------------------------------------------
# Module-level helpers called from FastAPI lifespan
# ---------------------------------------------------------------------------


def build_interface(checkpointer: BaseCheckpointSaver) -> TelegramInterface:
    return TelegramInterface(checkpointer)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _safe_filename(filename: str) -> str:
    """Return a filesystem-safe version of *filename*.

    Strips any path components (prevents traversal) and replaces characters
    that could cause issues on Linux filesystems. Truncates to 200 chars.
    """
    name = Path(filename).name  # strip any directory components
    safe = re.sub(r"[^\w.\-]", "_", name)
    return safe[:200] if safe else "file"


def _render_tool_lines(entries: list[_ToolEntry], spin_tick: int) -> str:
    """Render the rolling tool list as plain text.

    Truncates oldest *entries* (keeping each subagent header attached to its
    children) and caps the nested sub-lines per subagent block so a busy
    subagent can never push its own ``Agent()`` header out of view.
    """
    spinner = _SPINNER[spin_tick % len(_SPINNER)]
    visible = entries
    hidden = 0
    if len(entries) > _MAX_TOOL_ENTRIES:
        hidden = len(entries) - _MAX_TOOL_ENTRIES
        visible = entries[-_MAX_TOOL_ENTRIES:]

    lines: list[str] = [f"…+{hidden} more"] if hidden else []
    for entry in visible:
        args_part = f"({entry.args_preview})" if entry.args_preview else "()"
        if entry.done:
            icon = "🔴" if entry.error else "🟢"
        else:
            icon = spinner
        lines.append(f"{icon} {entry.label}{args_part}")
        if entry.sub_lines:
            subs = entry.sub_lines
            if len(subs) > _MAX_SUBLINES:
                lines.append(f"   ↳ …+{len(subs) - _MAX_SUBLINES} more")
                subs = subs[-_MAX_SUBLINES:]
            lines.extend(f"   ↳ {s}" for s in subs)
    return "\n".join(lines)


def _flood_wait(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception()
    return exc.retry_after if isinstance(exc, RetryAfter) else 2.0 * retry_state.attempt_number


@retry(
    retry=retry_if_exception_type((RetryAfter, TimedOut, NetworkError)),
    stop=stop_after_attempt(3),
    wait=_flood_wait,
    reraise=True,
)
async def _reply_text_with_retry(msg: Message, text: str, **kwargs) -> Message:
    """Send a reply with automatic retry on transient Telegram errors."""
    return await msg.reply_text(text, do_quote=False, **kwargs)


@retry(
    retry=retry_if_exception_type((RetryAfter, TimedOut, NetworkError)),
    stop=stop_after_attempt(3),
    wait=_flood_wait,
    reraise=True,
)
async def _send_attachment(msg, attachment: Attachment) -> None:
    data = io.BytesIO(attachment.data)
    caption = attachment.caption or None
    if isinstance(attachment, ImageAttachment):
        await msg.reply_photo(photo=data, caption=caption, disable_notification=True)
    elif isinstance(attachment, FileAttachment):
        data.name = attachment.filename
        await msg.reply_document(
            document=data, caption=caption, filename=attachment.filename, disable_notification=True
        )


async def _typing_loop(update: Update, stop_event: asyncio.Event) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
    while not stop_event.is_set():
        await update.get_bot().send_chat_action(
            chat_id=chat_id, action=ChatAction.TYPING, message_thread_id=topic_id
        )
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4.0)
        except TimeoutError:
            pass


_TELEGRAM_TAGS = {"b", "strong", "i", "em", "code", "pre", "a", "s", "u", "span", "blockquote"}
_BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th"}


class _TelegramSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._out: list[str] = []
        self._open: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _TELEGRAM_TAGS:
            attr_str = ""
            if tag == "a":
                href = next((v for k, v in attrs if k == "href"), None)
                if href:
                    attr_str = f' href="{href}"'
            elif tag == "span":
                cls = next((v for k, v in attrs if k == "class"), None)
                if cls:
                    attr_str = f' class="{cls}"'
            self._out.append(f"<{tag}{attr_str}>")
            self._open.append(tag)
        elif tag in _BLOCK_TAGS or tag == "br":
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _TELEGRAM_TAGS:
            if self._open and self._open[-1] == tag:
                self._out.append(f"</{tag}>")
                self._open.pop()
        elif tag in _BLOCK_TAGS:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        self._out.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._out.append(f"&#{name};")

    def result(self) -> str:
        for tag in reversed(self._open):
            self._out.append(f"</{tag}>")
        return re.sub(r"\n{3,}", "\n\n", "".join(self._out)).strip()


def _table_to_pre(match: re.Match) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), re.DOTALL)
    parsed = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)
        parsed.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])
    if not parsed or not parsed[0]:
        return ""
    col_count = len(parsed[0])
    widths = [max(len(r[i]) if i < len(r) else 0 for r in parsed) for i in range(col_count)]
    lines = []
    for i, row in enumerate(parsed):
        cells = row[:col_count] + [""] * (col_count - len(row))
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(cells)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "<pre>" + "\n".join(lines) + "</pre>"


def _to_html(text: str) -> str:
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    html = re.sub(r"<table[^>]*>.*?</table>", _table_to_pre, html, flags=re.DOTALL)
    sanitizer = _TelegramSanitizer()
    sanitizer.feed(html)
    return sanitizer.result()


def _draft_preview(text: str) -> str:
    """Return *text* trimmed to fit a single Telegram message, keeping the tail.

    Streaming previews (sendMessageDraft and live editMessageText) must never
    exceed the message length limit.  The final response is chunked separately
    via ``_chunk``, so showing only the most recent characters here is fine.
    """
    if len(text) <= _TG_MAX_LEN:
        return text
    return "…" + text[-(_TG_MAX_LEN - 1) :]


def _chunk(text: str) -> list[str]:
    """Split text into chunks that fit within Telegram's message length limit.

    Prefers splitting at paragraph boundaries, then line boundaries, then
    hard-cuts at the limit as a last resort.
    """
    if len(text) <= _TG_MAX_LEN:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > _TG_MAX_LEN:
        split = remaining.rfind("\n\n", 0, _TG_MAX_LEN)
        if split == -1:
            split = remaining.rfind("\n", 0, _TG_MAX_LEN)
        if split == -1:
            split = _TG_MAX_LEN
        chunks.append(remaining[:split].strip())
        remaining = remaining[split:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


_TG_THREAD_RE = re.compile(r"^tg-(-?\d+)-(?:topic-(\d+)|\d+)$")


def _parse_thread_id(thread_id: str) -> tuple[int, int | None]:
    """Parse a Telegram thread ID into ``(chat_id, topic_id | None)``.

    Accepts both DM format ``tg-{chat_id}-{session}`` and topic format
    ``tg-{chat_id}-topic-{topic_id}``.  Raises ValueError for unrecognised formats.
    """
    m = _TG_THREAD_RE.match(thread_id)
    if not m:
        raise ValueError(f"Unparseable Telegram thread ID: {thread_id!r}")
    chat_id = int(m.group(1))
    topic_id = int(m.group(2)) if m.group(2) is not None else None
    return chat_id, topic_id


def _forward_sender(msg: Message) -> str | None:
    """Return a human-readable sender name for a forwarded message, or None."""
    match msg.forward_origin:
        case None:
            return None
        case MessageOriginUser(sender_user=user):
            return user.full_name
        case MessageOriginHiddenUser(sender_user_name=name):
            return name
        case MessageOriginChat(sender_chat=chat):
            return chat.title or chat.username
        case MessageOriginChannel(chat=chat):
            return chat.title or chat.username
        case _:
            return None
