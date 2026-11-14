"""Telegram bot — polling mode.

Telegram is treated as just another frontend via TelegramInterface(BaseInterface[Update]).

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed. If TELEGRAM_BOT_TOKEN is not set the bot
is silently disabled.

Supported input types: text, voice (transcribed), audio, photos and albums, video, video
notes, animations, stickers, documents, location.
"""

import asyncio
import contextlib
import io
import logging
import re
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from html.parser import HTMLParser
from typing import Any

import markdown as md
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
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from aug.api.interfaces.base import (
    BaseInterface,
    ContentPart,
    IncomingMessage,
    LocationContent,
    TextContent,
    store_file,
)
from aug.api.interfaces.telegram.claude_login import _ClaudeLoginMixin
from aug.api.interfaces.telegram.ssh import _SshMixin
from aug.api.interfaces.telegram.utils import (
    escape,
    get_context_id,
    get_conversation_id,
    get_parent_conversation_id,
    is_allowed,
    parse_chat_conversation,
    restricted,
)
from aug.config import get_settings
from aug.core.harness import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
    FileEmitted,
    HarnessEvent,
    StatusUpdate,
    TextDelta,
    ToolEnded,
    ToolProgress,
    ToolStarted,
    TurnEnded,
    pending_approvals,
)
from aug.core.jobs import job_runner
from aug.core.prompts import USER_FILE_FAILED, USER_FILE_TOO_LARGE
from aug.core.registry import list_agents
from aug.core.run import run_registry
from aug.core.tools.approval import list_approvals, revoke_approval
from aug.core.tools.display import format_tool
from aug.utils.file_settings import load_settings
from aug.utils.media import guess_media_type
from aug.utils.skills import SKILLS_DIR, load_skills
from aug.utils.state import TelegramChatState, load_state, save_state

logger = logging.getLogger(__name__)

_SPINNER = ["🌑", "🌘", "🌗", "🌖", "🌕", "🌔", "🌓", "🌒"]
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
_MAX_TOOL_ENTRIES = 6  # max top-level tool entries shown before "…" truncation
_MAX_SUBLINES = 5  # max nested sub-tool lines shown per subagent block
# Tools with no status line: the file they send is the sign they ran.
_SILENT_TOOLS = {"respond_with_file"}
# In chats without draft support each update is a real editMessageText (rate-limited),
# unlike the free sendMessageDraft previews used in DMs.  Tool status refreshes slightly
# slower than the answer text since they may compete for the chat's edit budget.
_TOOL_EDIT_THROTTLE = 3.0
_ANSWER_EDIT_THROTTLE = 2.0

# Approval buttons carry ap:{request id}:{code}; the request itself stays server-side.
_APPROVAL_CODES = {
    "o": ApprovalDecision.APPROVED_ONCE,
    "a": ApprovalDecision.APPROVED_ALWAYS,
    "d": ApprovalDecision.DENIED,
}
# What an approval prompt becomes once it is answered, by decision.
_APPROVAL_OUTCOMES = {
    ApprovalDecision.APPROVED_ONCE: "✅ Approved for this run.",
    ApprovalDecision.APPROVED_ALWAYS: "✅ Approved — rule saved.",
    ApprovalDecision.DENIED: "❌ Command denied.",
    ApprovalDecision.REPLIED: "❌ Not run — you replied instead.",
    ApprovalDecision.TIMED_OUT: "⌛ Not run — the approval timed out.",
    ApprovalDecision.NO_USER: "❌ Not run — nobody was there to approve it.",
}
_APPROVAL_STOPPED = "⏹ Not run — stopped."
_FILE_SEND_FAILED = "⚠️ Couldn't send {filename}."
_PHOTO_TYPES = {"image/jpeg", "image/png", "image/webp"}
_APPROVAL_GONE = "⚠️ This approval has already been resolved."


@dataclass
class _ToolEntry:
    run_id: str
    label: str
    args_preview: str
    is_subagent: bool = False
    sub_lines: list[str] = field(default_factory=list)
    done: bool = False
    error: bool = False
    stopped: bool = False  # cut short by /stop


@dataclass(eq=False)
class _Album:
    context_id: str
    items: list[Update] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    timer: asyncio.Task[None] | None = None


class _AlbumCollector:
    """Holds an album's items until the album is complete.

    Telegram delivers an album as one update per item, sharing a media_group_id, with
    no marker on the last one.  Each new item restarts a short wait; once it passes
    with nothing new, the items go to *on_album* together, in the order they were sent.
    An item that arrives after that starts a new batch of its own.
    ``settled(context_id)`` waits until no album in that chat is still being collected
    or handed on, so a message sent with the album (a forward's text, "what's this?")
    can join it.
    """

    def __init__(
        self, on_album: Callable[[list[Update]], Awaitable[None]], wait: float = 1.0
    ) -> None:
        self._on_album = on_album
        self._wait = wait
        self._collecting: dict[str, _Album] = {}  # group → the batch still open
        self._handing: set[_Album] = set()  # batches closed, being handed on
        self._running: set[asyncio.Task[None]] = set()

    def add(self, update: Update, context_id: str) -> None:
        group = str(update.message.media_group_id)  # type: ignore[union-attr]
        album = self._collecting.get(group)
        if album is None:
            album = self._collecting[group] = _Album(context_id)
        album.items.append(update)
        if album.timer is not None:
            album.timer.cancel()
        album.timer = asyncio.create_task(self._fire(group, album))
        self._running.add(album.timer)
        album.timer.add_done_callback(self._running.discard)

    async def settled(self, context_id: str) -> None:
        """Return once every album in *context_id* has been handed on."""
        while pending := [
            album
            for album in (*self._collecting.values(), *self._handing)
            if album.context_id == context_id
        ]:
            await pending[0].done.wait()

    async def _fire(self, group: str, album: _Album) -> None:
        await asyncio.sleep(self._wait)
        # From here on this batch is closed: a straggler starts a new one.
        if self._collecting.get(group) is album:
            del self._collecting[group]
        self._handing.add(album)
        try:
            await self._on_album(sorted(album.items, key=lambda u: u.message.message_id))  # type: ignore[union-attr]
        except Exception:
            logger.exception("album handling failed group=%s", group)
        finally:
            self._handing.discard(album)
            album.done.set()


_TG_MAX_LEN = MessageLimit.MAX_TEXT_LENGTH
_SECRET_NAME, _SECRET_VALUE = range(2)
# Chats where sendMessageDraft is not supported — learned at runtime, reset on restart
_no_draft_chats: set[int] = set()


class TelegramInterface(_SshMixin, _ClaudeLoginMixin, BaseInterface[Update]):
    _debounce_window = 0.1  # 100ms — collects rapid-fire forwarded messages into one run

    def __init__(self) -> None:
        super().__init__()
        self._bot_app = None
        self._albums = _AlbumCollector(self._handle_album)

    # ------------------------------------------------------------------
    # BaseInterface implementation
    # ------------------------------------------------------------------

    async def receive_message(self, context: Update) -> IncomingMessage | None:
        # edited_message only reaches here from _handle_live_location — edited text
        # messages match no handler and are dropped before this point.
        msg = context.message or context.edited_message
        if not msg:
            return None
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        user_id = context.effective_user.id  # type: ignore[union-attr]
        if not is_allowed(user_id):
            return None

        context_id = get_context_id(chat_id, topic_id=msg.message_thread_id)

        parts: list[ContentPart] = []
        if media := _media_of(msg):
            parts.append(await _download(media))
            if msg.sticker and msg.sticker.emoji:
                parts.append(TextContent(text=f"[sticker: {msg.sticker.emoji}]"))
            if msg.caption:
                parts.append(TextContent(text=msg.caption))
        elif msg.location:
            parts.append(_location_content(msg))
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
            context_id=context_id,
            agent_version=self.get_agent_version(context_id),
            user_id=str(user_id),
        )

    async def send_stream(self, stream: AsyncIterator[HarnessEvent], context: Update) -> None:
        msg = context.effective_message
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        bot = msg.get_bot()  # type: ignore[union-attr]

        stop_typing = asyncio.Event()
        approval_prompts: dict[str, Message] = {}  # request id → its prompt, until answered
        typing_task = asyncio.create_task(
            _typing_loop(context, stop_typing, paused=lambda: bool(approval_prompts))
        )

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
        turn_failed = False
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
                # Waiting on the user can take minutes: no animation (and no edits
                # against the chat's rate limit) until they answer.
                if any(not e.done for e in tool_entries) and not approval_prompts:
                    await _push_tool_draft()

        spinner_task = asyncio.create_task(_spinner_loop())

        try:
            async for event in stream:
                match event:
                    case TextDelta(text=delta, parent_id=None) if delta:
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

                    case ToolStarted(name=sub_tool, args=sub_args, parent_id=parent_id) if (
                        parent_id is not None
                    ):
                        # A subagent's tool call: a line nested under the subagent's entry.
                        parent_entry = run_id_to_entry.get(parent_id)
                        if parent_entry and sub_tool not in _SILENT_TOOLS:
                            sub_label, sub_preview = format_tool(sub_tool, sub_args)
                            inner = f"({sub_preview})" if sub_preview else "()"
                            parent_entry.sub_lines.append(f"{sub_label}{inner}")
                            await _push_tool_draft()

                    case ToolStarted(id=run_id, name=tool_name, args=args):
                        # Commit the text so far first: it must land before any file.
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
                        if tool_name in _SILENT_TOOLS:
                            continue

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

                    case ToolProgress(text=step, parent_id=parent_id) if step:
                        parent_entry = run_id_to_entry.get(parent_id) if parent_id else None
                        if parent_entry:
                            line = step.split("\n", 1)[0].strip()
                            if len(line) > 60:
                                line = line[:60] + "…"
                            parent_entry.sub_lines.append(line)
                            await _push_tool_draft()

                    case ToolEnded(parent_id=parent_id) if parent_id is not None:
                        pass  # a subagent's tool: its line stays as it is

                    case ToolEnded(id=run_id, error=error):
                        entry = run_id_to_entry.get(run_id)
                        if entry:
                            entry.done = True
                            entry.error = error
                            await _push_tool_draft()

                    case FileEmitted() as file:
                        # Sent as soon as it is made, a subagent's included.
                        topic_id = msg.message_thread_id  # type: ignore[union-attr]
                        await _deliver_file(bot, chat_id, topic_id, file, silent=True)

                    case ApprovalRequested(id=approval_id, request=request):
                        prompt = await self._send_approval_prompt(msg, approval_id, request)  # type: ignore[arg-type]
                        if prompt is not None:
                            approval_prompts[approval_id] = prompt

                    case ApprovalResolved(id=approval_id, decision=decision):
                        prompt = approval_prompts.pop(approval_id, None)
                        if prompt is not None:
                            await _close_approval_prompt(prompt, _APPROVAL_OUTCOMES[decision])

                    case StatusUpdate(text=text) if text:
                        try:
                            await msg.reply_text(  # type: ignore[union-attr]
                                text,
                                disable_notification=True,
                                do_quote=False,
                            )
                        except Exception:
                            logger.debug("status event send failed", exc_info=True)

                    case TurnEnded(error=error) if error is not None:
                        turn_failed = True

                    case TurnEnded(cancelled=True):
                        for entry in tool_entries:
                            if not entry.done:
                                entry.done = entry.stopped = True

            if turn_failed:
                return  # the caller reports the error; `finally` marks open tools failed

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
            # Requests the turn never answered (/stop, an error): no buttons left to tap.
            for prompt in approval_prompts.values():
                await _close_approval_prompt(prompt, _APPROVAL_STOPPED)
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

    async def resolve_context(
        self,
        context_id: str,
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
        if context_id == "default" or context_id.startswith("default:"):
            if context_id.startswith("default:"):
                cid = int(context_id.split(":", 1)[1])
                return get_context_id(cid, None)
            if chat_id is not None:
                return get_context_id(chat_id, None)
            # Legacy fallback: tasks stored before the "default:{chat_id}" encoding.
            for conversation in load_settings().conversations:
                cid = parse_chat_conversation(conversation)
                if cid is not None and cid > 0:
                    logger.warning(
                        "resolve_context: using first known chat_id %d as 'default' — "
                        "tasks should store 'default:{chat_id}' for deterministic routing",
                        cid,
                    )
                    return get_context_id(cid, None)
            raise ValueError("No default Telegram DM chat found in settings")

        if context_id == "new":
            if chat_id is None:
                raise ValueError("chat_id is required when context_id='new'")
            if not self._bot_app:
                raise RuntimeError("Telegram bot is not running")
            topic = await self._bot_app.bot.create_forum_topic(
                chat_id=chat_id,
                name=topic_name or "New Topic",
            )
            return get_context_id(chat_id, topic_id=topic.message_thread_id)

        return context_id

    def conversation_id(self, context_id: str) -> str:
        return get_conversation_id(context_id)

    def parent_conversation_id(self, conversation_id: str) -> str | None:
        return get_parent_conversation_id(conversation_id)

    async def send_proactive(self, context_id: str, text: str) -> None:
        """Send *text* to *context_id* without an agent turn.

        Used for ``type="forward"`` external pushes that relay a message
        verbatim (e.g. a Home Assistant notification).
        """
        if not self._bot_app:
            raise RuntimeError("Telegram bot is not running")
        chat_id, topic_id = _parse_context_id(context_id)
        for chunk in _chunk(text):
            await self._bot_app.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=topic_id,
            )

    async def announcement_threads(self) -> list[str]:
        """Return the current DM thread of every user allowed to use the bot.

        Service announcements go to people rather than to rooms: the allow-list is
        the set of owners, and a Telegram user ID is also their DM chat ID.  When
        no allow-list is configured (any chat may talk to the bot) there is no
        owner to address, so the DM chats already seen in settings are used.
        Group and forum chats — negative IDs — are never announced to.
        """
        chat_ids = {cid for cid in get_settings().allowed_chat_ids if cid > 0}
        if not chat_ids:
            known = (parse_chat_conversation(c) for c in load_settings().conversations)
            chat_ids = {cid for cid in known if cid is not None and cid > 0}
        return [get_context_id(cid, None) for cid in sorted(chat_ids)]

    async def send_proactive_stream(
        self, context_id: str, stream: AsyncIterator[HarnessEvent]
    ) -> None:
        """Stream the agent response to *context_id* using draft updates, then send final message.

        Uses sendMessageDraft to show tokens as they arrive (same as interactive chat).
        Falls back to send-then-edit if drafts are unsupported. Tool call messages are
        suppressed; only the final text and attachments are delivered.
        Used for ``type="agent"`` pushes and scheduled tasks.
        """
        if not self._bot_app:
            raise RuntimeError("Telegram bot is not running")
        chat_id, topic_id = _parse_context_id(context_id)
        bot = self._bot_app.bot

        accumulated_text = ""
        files: list[FileEmitted] = []
        use_draft = chat_id not in _no_draft_chats
        last_draft_time = 0.0
        stream_msg = None
        last_stream_edit = 0.0

        async for event in stream:
            match event:
                case TextDelta(text=delta, parent_id=None) if delta:
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
                case FileEmitted() as file:
                    files.append(file)

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

        # After the text: a push reads as a message with its files attached.
        for file in files:
            await _deliver_file(bot, chat_id, topic_id, file, silent=False)

    async def _send_approval_prompt(
        self, msg: Message, approval_id: str, request: ApprovalRequest
    ) -> Message | None:
        """Ask the user to approve *request*, with buttons that answer *approval_id*."""
        buttons = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Run Once", callback_data=f"ap:{approval_id}:o")],
                [InlineKeyboardButton("✅✅ Allow Always", callback_data=f"ap:{approval_id}:a")],
                [InlineKeyboardButton("❌ Deny", callback_data=f"ap:{approval_id}:d")],
            ]
        )
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
            return await msg.reply_text(
                text,
                parse_mode="HTML",
                reply_markup=buttons,
                link_preview_options=_NO_PREVIEW,
                do_quote=False,
            )
        except Exception:
            logger.warning("approval_prompt_html_failed, retrying plain", exc_info=True)
        try:
            return await msg.reply_text(
                f"⚠️ Approval required\n{request.description}",
                reply_markup=buttons,
                do_quote=False,
            )
        except Exception:
            # Nobody can answer it now; the tool gets TIMED_OUT in the end.
            logger.exception("approval_prompt_failed id=%s", approval_id)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_bot(self) -> Application:
        builder = Application.builder().token(get_settings().TELEGRAM_BOT_TOKEN)  # type: ignore[arg-type]
        base_url = get_settings().TELEGRAM_BASE_URL
        if base_url:
            builder = builder.base_url(f"{base_url}/bot").base_file_url(f"{base_url}/file/bot")
        bot_app = builder.concurrent_updates(True).build()
        bot_app.add_handler(CommandHandler("clear", self._handle_clear))
        bot_app.add_handler(CommandHandler("stop", self._handle_stop))
        bot_app.add_handler(CommandHandler("context_id", self._handle_context_id))
        bot_app.add_handler(CommandHandler("throttle", self._handle_throttle))
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
            CallbackQueryHandler(self._handle_approval_callback, pattern=r"^(ap|approval):")
        )
        bot_app.add_handler(
            CallbackQueryHandler(
                self._handle_approvals_revoke_callback, pattern=r"^approval_revoke:"
            )
        )
        bot_app.add_handler(CommandHandler("approvals", self._handle_approvals))
        self.build_ssh_handlers(bot_app)
        self.build_claude_login_handlers(bot_app)
        bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        bot_app.add_handler(MessageHandler(filters.VOICE, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.AUDIO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.PHOTO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.VIDEO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.VIDEO_NOTE, self._handle_input))
        # Before documents: an animation message carries a document too.
        bot_app.add_handler(MessageHandler(filters.ANIMATION, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.Document.ALL, self._handle_input))
        bot_app.add_handler(
            MessageHandler(
                filters.LOCATION & filters.UpdateType.EDITED_MESSAGE, self._handle_live_location
            )
        )
        bot_app.add_handler(
            MessageHandler(
                filters.LOCATION & ~filters.UpdateType.EDITED_MESSAGE, self._handle_input
            )
        )
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
                ("claude_login", "Sign in to Claude for the Claude agents"),
                ("prompt", "Export current system prompt as a file"),
                ("consolidate", "Run memory consolidation now"),
                ("consolidate_deep", "Run deep (weekly) memory consolidation now"),
                ("approvals", "List and revoke saved SSH command approvals"),
                ("ssh", "Manage SSH targets"),
                ("context_id", "Show the context ID (current session) of this chat"),
                ("throttle", "Set live location update throttle (seconds)"),
            ]
        )
        await self._bot_app.start()
        await self._bot_app.updater.start_polling()
        logger.info("Telegram bot started (polling).")

    async def stop_polling(self, app) -> None:
        """Gracefully stop the bot. Called from FastAPI lifespan shutdown."""
        if not self._bot_app:
            return
        await self.end_claude_logins()
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
        if update.message.media_group_id:
            context_id = get_context_id(
                update.effective_chat.id,  # type: ignore[union-attr]
                topic_id=update.message.message_thread_id,
            )
            self._albums.add(update, context_id)
            return
        await self.run(update)

    async def _handle_album(self, updates: list[Update]) -> None:
        """An album's items, received together, become one message and one turn.

        An item that fails to download becomes a note; the rest still arrive.
        """
        received = await asyncio.gather(
            *(self.receive_message(u) for u in updates), return_exceptions=True
        )
        incoming = [m for m in received if isinstance(m, IncomingMessage)]
        if not incoming:
            for error in (r for r in received if isinstance(r, BaseException)):
                logger.error("album item failed", exc_info=error)
            return
        parts: list[ContentPart] = []
        for update, result in zip(updates, received, strict=True):
            if isinstance(result, BaseException):
                logger.warning("album item failed", exc_info=result)
                media = _media_of(update.message)  # type: ignore[arg-type]
                name = media.filename if media else "a file"
                parts.append(TextContent(text=USER_FILE_FAILED.format(filename=name)))
            elif result is not None:
                # Each item repeats a forward's "[Forwarded from …]"; keep it once.
                parts.extend(p for p in result.parts if p not in parts)
        await self.handle(incoming[0].model_copy(update={"parts": parts}), updates[0])

    async def _before_turn(self, context_id: str) -> None:
        # A message sent with an album waits for it, so both make one turn.
        await self._albums.settled(context_id)

    async def _handle_live_location(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Live location update — an edited_message carrying a new position.

        Telegram edits the original message roughly once a minute for the whole
        live_period. The coordinates are persisted unless they arrive out of order;
        the agent is only woken when a run is already going (cheap injection) or the
        conversation's throttle has elapsed.
        """
        msg = update.edited_message
        if not msg or not msg.location:
            return
        if not update.effective_user or not is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        context_id = get_context_id(chat_id, topic_id=msg.message_thread_id)
        if not self.record_location(
            context_id, str(update.effective_user.id), _location_content(msg)
        ):
            return  # delivered out of order — we already hold a newer position

        # Cheap gate: skip the whole pipeline for an update nothing is waiting on.
        # It is only a gate — run() takes the real decision under its thread lock,
        # since the run seen here can finish while run() is still preprocessing.
        injecting = run_registry.accepting(context_id)
        if not injecting and not self.claim_location_run(context_id):
            logger.debug("live location throttled for %s", context_id)
            return

        # run() injects into the active run if there is one, else starts a new one.
        await self.run(update)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    @restricted
    async def _handle_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
        stopped = await self.stop_run(get_context_id(chat_id, topic_id))
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
        st.telegram.chats.setdefault(str(chat_id), TelegramChatState()).session += 1
        save_state(st)
        await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]

    @restricted
    async def _handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
        current = self.get_agent_version(get_context_id(chat_id, topic_id))
        current_label = current if current in list_agents() else "none selected"
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
            f"Current version: <code>{escape(current_label)}</code>\nChoose a version:",
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
        # An inaccessible message (older than ~48h) carries no message_thread_id, so we
        # cannot tell which topic the button belongs to.  Guessing would write the
        # selection to the wrong conversation — exactly the leak this feature prevents —
        # and edit_message_text would raise on it anyway.  Answer exactly once on every
        # path: Telegram rejects a second answer for the same callback query.
        msg = update.effective_message
        if msg is None or update.effective_chat is None:
            await query.answer("That menu is too old — run /version again.", show_alert=True)
            return
        await query.answer()
        chat_id = update.effective_chat.id
        agent_name = query.data.split(":", 1)[1]  # type: ignore[union-attr]
        if agent_name not in list_agents():
            await query.edit_message_text("Unknown version.")
            return
        self.set_agent_version(get_context_id(chat_id, msg.message_thread_id), agent_name)
        await query.edit_message_text(
            f"Switched to <code>{escape(agent_name)}</code>.", parse_mode="HTML"
        )

    @restricted
    async def _handle_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        all_skills = await asyncio.to_thread(load_skills)
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
        context_id = get_context_id(chat_id, update.effective_message.message_thread_id)  # type: ignore[union-attr]
        prompt_text = await self.session_prompt(context_id, "telegram")
        if not prompt_text:
            await update.message.reply_text("This chat's agent has no system prompt.")  # type: ignore[union-attr]
            return
        file = io.BytesIO(prompt_text.encode())
        file.name = "system_prompt.txt"
        await update.message.reply_document(document=file, filename="system_prompt.txt")  # type: ignore[union-attr]

    @restricted
    async def _handle_context_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.message.message_thread_id  # type: ignore[union-attr]
        context_id = get_context_id(chat_id, topic_id)
        await update.message.reply_text(  # type: ignore[union-attr]
            f"<code>{escape(context_id)}</code>",
            parse_mode="HTML",
        )

    @restricted
    async def _handle_throttle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
        context_id = get_context_id(chat_id, topic_id)
        reply = update.effective_message.reply_text  # type: ignore[union-attr]

        if not context.args:
            await reply(f"Live location throttle: {self.location_throttle(context_id)}s")
            return

        try:
            seconds = int(context.args[0])
        except ValueError:
            await reply("Usage: /throttle <seconds>")
            return
        if seconds < 60:
            await reply("Minimum throttle is 60 seconds.")
            return

        self.set_location_throttle(context_id, seconds)
        await reply(f"Live location throttle set to {seconds}s.")

    @restricted
    async def _handle_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
        context_id = get_context_id(chat_id, topic_id)
        await update.message.reply_text("🗜 Compacting conversation…")  # type: ignore[union-attr]
        try:
            agent_version = self.get_agent_version(context_id)
            if agent_version not in list_agents():
                await update.message.reply_text(  # type: ignore[union-attr]
                    "No agent selected. Use /version to pick one first."
                )
                return
            ran = await self._execute_compact(context_id, agent_version)
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
            ran = await job_runner.run("dream_light")
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
            await job_runner.run("dream_deep")
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
        """Handle user tapping Run Once / Allow Always / Deny on an approval prompt.

        The turn's own stream rewrites the prompt once the waiting tool has the answer.
        """
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        chat_id = update.effective_chat.id  # type: ignore[union-attr]

        # callback_data: "ap:{approval id}:{o|a|d}"
        parts = (query.data or "").split(":")
        decision = _APPROVAL_CODES.get(parts[2]) if len(parts) == 3 else None
        pending = pending_approvals.get(parts[1]) if decision is not None else None
        if pending is None:
            # Already answered, timed out, stopped, or asked before a restart.
            await query.edit_message_text(_APPROVAL_GONE)
            return
        if not pending.context_id.startswith(f"tg-{chat_id}-"):
            await query.edit_message_text("❌ Unauthorized.")
            return
        if not pending_approvals.resolve(pending.id, decision):  # type: ignore[arg-type]
            await query.edit_message_text(_APPROVAL_GONE)

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


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


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
        if entry.stopped:
            icon = "⏹"
        elif entry.done:
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


async def _close_approval_prompt(prompt: Message, text: str) -> None:
    """Replace an approval prompt (and its buttons) with how it ended."""
    try:
        await prompt.edit_text(text)
    except Exception:
        logger.debug("approval prompt edit failed", exc_info=True)


async def _deliver_file(
    bot, chat_id: int, topic_id: int | None, file: FileEmitted, *, silent: bool
) -> None:
    """Send *file*; if that fails, tell the chat instead of dropping it silently."""
    try:
        await _send_file(bot, chat_id, topic_id, file, silent=silent)
    except Exception:
        logger.warning(
            "file send failed id=%s type=%s", file.file_id, file.media_type, exc_info=True
        )
        with contextlib.suppress(Exception):
            await bot.send_message(
                chat_id=chat_id,
                text=_FILE_SEND_FAILED.format(filename=file.filename),
                message_thread_id=topic_id,
            )


async def _send_file(
    bot, chat_id: int, topic_id: int | None, file: FileEmitted, *, silent: bool
) -> None:
    """Send *file* the way its media type shows best: as a photo, voice note, audio,
    video, animation or document.  A file Telegram refuses in that form (a photo too
    large, say) goes again as a document."""
    data = await asyncio.to_thread(file.path.read_bytes)
    method, field = _send_method(file.media_type)
    kwargs = {
        "chat_id": chat_id,
        "caption": file.caption[: MessageLimit.CAPTION_LENGTH] or None,
        "message_thread_id": topic_id,
        "disable_notification": silent,
        "filename": file.filename,
    }
    try:
        await _upload(getattr(bot, method), **{field: data}, **kwargs)
    except BadRequest:
        if method == "send_document":
            raise
        logger.info("%s refused id=%s, sending as a document", method, file.file_id)
        await _upload(bot.send_document, document=data, **kwargs)


@retry(
    # BadRequest is a NetworkError too, but resending what Telegram refused never helps;
    # nor is a timed-out upload retried, as Telegram may have received it all the same.
    retry=retry_if_exception_type((RetryAfter, NetworkError))
    & retry_if_not_exception_type((BadRequest, TimedOut)),
    stop=stop_after_attempt(3),
    wait=_flood_wait,
    reraise=True,
)
async def _upload(send: Callable[..., Awaitable[Message]], **kwargs) -> None:
    await send(**kwargs)


def _send_method(media_type: str) -> tuple[str, str]:
    """The Bot API method for *media_type*, and its file parameter."""
    if media_type in _PHOTO_TYPES:
        return "send_photo", "photo"
    if media_type == "image/gif":
        return "send_animation", "animation"
    if media_type in ("audio/ogg", "audio/opus"):
        return "send_voice", "voice"
    if media_type.startswith("audio/"):
        return "send_audio", "audio"
    if media_type in ("video/mp4", "video/quicktime"):
        return "send_video", "video"
    return "send_document", "document"


async def _typing_loop(
    update: Update, stop_event: asyncio.Event, paused: Callable[[], bool] = lambda: False
) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    topic_id = update.effective_message.message_thread_id  # type: ignore[union-attr]
    while not stop_event.is_set():
        if not paused():
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


def _parse_context_id(context_id: str) -> tuple[int, int | None]:
    """Parse a Telegram thread ID into ``(chat_id, topic_id | None)``.

    Accepts both DM format ``tg-{chat_id}-{session}`` and topic format
    ``tg-{chat_id}-topic-{topic_id}``.  Raises ValueError for unrecognised formats.
    """
    m = _TG_THREAD_RE.match(context_id)
    if not m:
        raise ValueError(f"Unparseable Telegram thread ID: {context_id!r}")
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


def _location_content(msg: Message) -> LocationContent:
    """Translate a Telegram location message into the interface-agnostic LocationContent."""
    location = msg.location
    # live_period is int seconds today, timedelta under PTB_TIMEDELTA=true.
    period = location.live_period  # type: ignore[union-attr]
    if isinstance(period, timedelta):
        period = int(period.total_seconds())
    # date stays at the original send time across edits, so it anchors the expiry;
    # edit_date advances with each update, so it orders them.
    reported = msg.edit_date or msg.date
    return LocationContent(
        latitude=location.latitude,  # type: ignore[union-attr]
        longitude=location.longitude,  # type: ignore[union-attr]
        live_period=period,
        sent_at=msg.date.timestamp() if msg.date else None,
        reported_at=reported.timestamp() if reported else None,
        sender_name=msg.from_user.full_name if msg.from_user else None,
    )


@dataclass(frozen=True)
class _Media:
    """A file in a Telegram message: the object to download it from, and what it is."""

    source: Any  # PhotoSize, Voice, Audio, Video, VideoNote, Animation, Sticker, Document
    filename: str
    mime_type: str
    transcribe: bool = False


def _media_of(msg: Message) -> _Media | None:
    """The file *msg* carries, if any.  Animation comes before document: an animation
    message carries both."""
    if msg.voice:
        return _Media(msg.voice, "voice.ogg", msg.voice.mime_type or "audio/ogg", transcribe=True)
    if msg.audio:
        return _Media(
            msg.audio, msg.audio.file_name or "audio.mp3", msg.audio.mime_type or "audio/mpeg"
        )
    if msg.photo:
        return _Media(msg.photo[-1], "photo.jpg", "image/jpeg")
    if msg.video:
        return _Media(
            msg.video, msg.video.file_name or "video.mp4", msg.video.mime_type or "video/mp4"
        )
    if msg.video_note:
        return _Media(msg.video_note, "video_note.mp4", "video/mp4")
    if msg.animation:
        name = msg.animation.file_name or "animation.mp4"
        return _Media(msg.animation, name, msg.animation.mime_type or "video/mp4")
    if msg.sticker:
        if msg.sticker.is_video:
            return _Media(msg.sticker, "sticker.webm", "video/webm")
        if msg.sticker.is_animated:
            return _Media(msg.sticker, "sticker.tgs", "application/x-tgsticker")
        return _Media(msg.sticker, "sticker.webp", "image/webp")
    if msg.document:
        name = msg.document.file_name or "document"
        mime = msg.document.mime_type
        if not mime or mime == "application/octet-stream":
            mime = guess_media_type(name)
        return _Media(msg.document, name, mime)
    return None


async def _download(media: _Media) -> ContentPart:
    """Download *media* into the inbox.  A file over the Bot API's download limit
    becomes a note saying so, so the agent can tell the user."""
    try:
        tg_file = await media.source.get_file()
    except BadRequest as e:
        if "too big" not in str(e).lower():
            raise
        logger.info("telegram file too big to download name=%s", media.filename)
        return TextContent(text=USER_FILE_TOO_LARGE.format(filename=media.filename))
    data = await tg_file.download_as_bytearray()  # bytes-like; no second copy
    return await store_file(data, media.filename, media.mime_type, transcribe=media.transcribe)
