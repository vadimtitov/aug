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
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

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
from telegram.error import BadRequest, RetryAfter
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

from aug.api.interfaces.base import (
    BaseInterface,
    FileContent,
    IncomingMessage,
    LocationContent,
    TextContent,
)
from aug.api.interfaces.telegram.ssh import _SshMixin
from aug.api.interfaces.telegram.utils import _escape, _is_allowed, _restricted, _thread_id
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
from aug.core.tools.output import Attachment, FileAttachment, ImageAttachment, ToolOutput
from aug.utils.data import UPLOADS_DIR
from aug.utils.skills import SKILLS_DIR, load_skills
from aug.utils.state import get_state, set_state
from aug.utils.user_settings import get_setting, set_setting

logger = logging.getLogger(__name__)

_SPINNER = ["🌑", "🌘", "🌗", "🌖", "🌕", "🌔", "🌓", "🌒"]
_TOOL_NAMES = {
    "brave_search": "Search",
    "fetch_page": "Fetch",
    "run_bash": "Bash",
    "browser": "Browser",
    "note": "Note",
    "gmail_search": "Gmail",
    "gmail_read_thread": "Gmail",
    "gmail_send": "Send email",
    "gmail_draft": "Draft email",
    "respond_with_file": "Send file",
    "generate_image": "Generate image",
    "edit_image": "Edit image",
    "portainer_list_containers": "Portainer",
    "portainer_container_logs": "Portainer logs",
    "portainer_container_action": "Portainer action",
    "portainer_list_stacks": "Portainer stacks",
    "portainer_deploy_stack": "Portainer deploy",
    "portainer_stack_action": "Portainer stack",
    "run_ssh": "SSH",
    "list_ssh_targets": "SSH targets",
    "set_reminder": "Set reminder",
    "get_skill": "Get skill",
    "save_skill": "Save skill",
    "write_skill_file": "Write skill file",
    "delete_skill": "Delete skill",
}
_ARG_TRUNCATE = 50
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
_TG_MAX_LEN = MessageLimit.MAX_TEXT_LENGTH
_SECRET_NAME, _SECRET_VALUE = range(2)


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
        if not _is_allowed(chat_id):
            return None

        thread_id = _thread_id(chat_id)
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
            agent_version=get_setting(
                "telegram", "chats", str(chat_id), "agent", default="default"
            ),
        )

    async def send_stream(self, stream: AsyncIterator[AgentEvent], context: Update) -> None:
        msg = context.effective_message
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_typing_loop(context, stop_typing))

        accumulated_text = ""
        stream_msg = None
        last_stream_edit = 0.0
        # tool_run_id → (tool_name, args, tool_msg, task, step_holder)
        # step_holder is a 1-element list so the spinner and progress
        # handler share the same mutable ref
        tool_msgs: dict[str, tuple[str, dict, object, asyncio.Task, list]] = {}
        # parent_id → tool_run_id (for matching ToolProgressEvent to its tool_msg)
        progress_index: dict[str, str] = {}
        stream_completed_normally = False

        try:
            async for event in stream:
                match event:
                    case ChatModelStreamEvent(delta=delta) if delta:
                        accumulated_text += delta
                        now = asyncio.get_running_loop().time()
                        if stream_msg is None:
                            stream_msg = await msg.reply_text(  # type: ignore[union-attr]
                                accumulated_text,
                                link_preview_options=_NO_PREVIEW,
                                disable_notification=True,
                            )
                            last_stream_edit = now
                        elif now - last_stream_edit >= 0.3 and len(accumulated_text) <= _TG_MAX_LEN:
                            try:
                                await stream_msg.edit_text(
                                    accumulated_text, link_preview_options=_NO_PREVIEW
                                )
                                last_stream_edit = now
                            except RetryAfter as e:
                                last_stream_edit = now + e.retry_after
                            except Exception:
                                pass
                    case ToolStartEvent(
                        run_id=run_id, tool_name=tool_name, args=args, parent_ids=parent_ids
                    ):
                        text = _format_tool_call(tool_name, args, done=False)
                        try:
                            tool_msg = await msg.reply_text(  # type: ignore[union-attr]
                                text,
                                parse_mode="HTML",
                                link_preview_options=_NO_PREVIEW,
                                disable_notification=True,
                            )
                            step_holder: list[str] = [""]
                            spin = asyncio.create_task(
                                _spinner_task(tool_msg, tool_name, args, step_holder)
                            )
                            tool_msgs[run_id] = (tool_name, args, tool_msg, spin, step_holder)
                            for pid in parent_ids:
                                progress_index[pid] = run_id
                        except RetryAfter:
                            pass  # flood control — skip status message, run continues
                        if stream_msg is not None and accumulated_text:
                            try:
                                await stream_msg.edit_text(
                                    _to_html(accumulated_text),
                                    parse_mode="HTML",
                                    link_preview_options=_NO_PREVIEW,
                                )
                            except Exception:
                                pass
                        accumulated_text = ""
                        stream_msg = None
                    case ToolProgressEvent(parent_ids=parent_ids, step=step) if step:
                        tool_run_id = next(
                            (progress_index[pid] for pid in parent_ids if pid in progress_index),
                            None,
                        )
                        if tool_run_id and tool_run_id in tool_msgs:
                            tool_name, args, tool_msg, _, step_holder = tool_msgs[tool_run_id]
                            step_holder[0] = step
                    case StatusEvent(text=text) if text:
                        try:
                            await msg.reply_text(  # type: ignore[union-attr]
                                text,
                                disable_notification=True,
                            )
                        except Exception:
                            pass
                    case ToolEndEvent(
                        run_id=run_id, tool_name=tool_name, output=output, error=error
                    ):
                        entry = tool_msgs.pop(run_id, None)
                        progress_index = {
                            pid: rid for pid, rid in progress_index.items() if rid != run_id
                        }
                        if entry:
                            tool_name, args, tool_msg, task, _ = entry
                            task.cancel()
                            try:
                                await tool_msg.edit_text(  # type: ignore[union-attr]
                                    _format_tool_call(tool_name, args, done=True, error=error),
                                    parse_mode="HTML",
                                    link_preview_options=_NO_PREVIEW,
                                )
                            except Exception:
                                pass
                            if isinstance(output, ToolOutput) and output.attachments:
                                for attachment in output.attachments:
                                    await _send_attachment(msg, attachment)  # type: ignore[arg-type]

            if accumulated_text:
                chunks = _chunk(accumulated_text)
                for i, chunk in enumerate(chunks):
                    html = _to_html(chunk)
                    is_last = i == len(chunks) - 1
                    try:
                        if i == 0 and stream_msg is not None:
                            # Delete the silent intermediate message and send a fresh
                            # one so the user gets exactly one notification per response.
                            try:
                                await stream_msg.delete()  # type: ignore[union-attr]
                            except Exception:
                                pass
                            stream_msg = None
                        await msg.reply_text(  # type: ignore[union-attr]
                            html,
                            parse_mode="HTML",
                            link_preview_options=_NO_PREVIEW,
                            disable_notification=not is_last,
                        )
                    except RetryAfter as e:
                        await asyncio.sleep(e.retry_after)
                        await msg.reply_text(  # type: ignore[union-attr]
                            html,
                            parse_mode="HTML",
                            link_preview_options=_NO_PREVIEW,
                            disable_notification=not is_last,
                        )
                    except BadRequest:
                        logger.warning("Failed to send HTML, falling back to plain", exc_info=True)
                        await msg.reply_text(  # type: ignore[union-attr]
                            chunk,
                            link_preview_options=_NO_PREVIEW,
                            disable_notification=not is_last,
                        )

            stream_completed_normally = True

        finally:
            stop_typing.set()
            typing_task.cancel()
            for tool_name, args, tool_msg, task, _ in tool_msgs.values():
                task.cancel()
                try:
                    if stream_completed_normally:
                        # Stream ended cleanly (e.g. approval interrupt) — delete the
                        # pending tool message rather than showing it as errored.
                        await tool_msg.delete()  # type: ignore[union-attr]
                    else:
                        await tool_msg.edit_text(
                            _format_tool_call(tool_name, args, done=True, error=True),
                            parse_mode="HTML",
                            link_preview_options=_NO_PREVIEW,
                        )
                except Exception:
                    pass

    async def send_message(self, message: str, context: Update) -> None:
        for chunk in _chunk(message):
            await context.effective_message.reply_text(chunk)  # type: ignore[union-attr]

    async def request_approval(self, request: ApprovalRequest, context: Update) -> None:
        """Send an approval prompt with inline buttons to the user."""
        msg = context.effective_message  # type: ignore[union-attr]
        chat_id = context.effective_chat.id  # type: ignore[union-attr]
        thread_id = _thread_id(chat_id)
        agent_version = get_setting("telegram", "chats", str(chat_id), "agent", default="default")
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
            f"<b>Target:</b> <code>{_escape(request.resource)}</code>\n" if request.resource else ""
        )
        text = (
            f"⚠️ <b>Approval required</b>\n\n"
            f"<b>Tool:</b> <code>{_escape(request.tool_name)}</code>\n"
            f"{resource_line}"
            f"<b>Operation:</b>\n<pre>{_escape(request.operation)}</pre>"
        )
        try:
            await msg.reply_text(  # type: ignore[union-attr]
                text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(buttons),
                link_preview_options=_NO_PREVIEW,
            )
        except Exception:
            logger.warning("approval_prompt_html_failed, retrying plain", exc_info=True)
            await msg.reply_text(  # type: ignore[union-attr]
                f"⚠️ Approval required\n{request.description}",
                reply_markup=InlineKeyboardMarkup(buttons),
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

    async def send_notification(self, target_id: str, text: str) -> None:
        if not self._bot_app:
            raise RuntimeError("Telegram bot is not running")
        await self._bot_app.bot.send_message(chat_id=int(target_id), text=text)

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

    @_restricted
    async def _handle_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        stopped = self.stop_run(_thread_id(chat_id))
        msg = "Stopping..." if stopped else "Nothing is running."
        await update.message.reply_text(msg)  # type: ignore[union-attr]

    @_restricted
    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        current = get_state("telegram", "chats", str(chat_id), "session", default=0)
        set_state("telegram", "chats", str(chat_id), "session", value=current + 1)
        await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]

    @_restricted
    async def _handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        current = get_setting("telegram", "chats", str(chat_id), "agent", default="default")
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
            f"Current version: <code>{_escape(current)}</code>\nChoose a version:",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    @_restricted
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
        set_setting("telegram", "chats", str(chat_id), "agent", value=agent_name)
        await query.edit_message_text(
            f"Switched to <code>{_escape(agent_name)}</code>.", parse_mode="HTML"
        )

    @_restricted
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

    @_restricted
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
            f"Sending {len(files)} file(s) for skill <code>{_escape(skill_name)}</code>:",
            parse_mode="HTML",
        )
        msg = update.effective_message
        for f in files:
            file_bytes = await asyncio.to_thread(f.read_bytes)
            rel = f.relative_to(skill_dir)
            await msg.reply_document(  # type: ignore[union-attr]
                document=io.BytesIO(file_bytes), filename=str(rel)
            )

    @_restricted
    async def _handle_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        state = AgentState(
            messages=[],
            thread_id=_thread_id(chat_id),
            interface="telegram",
        )
        prompt_text = build_system_prompt(state)
        file = io.BytesIO(prompt_text.encode())
        file.name = "system_prompt.txt"
        await update.message.reply_document(document=file, filename="system_prompt.txt")  # type: ignore[union-attr]

    @_restricted
    async def _handle_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        agent_version = get_setting("telegram", "chats", str(chat_id), "agent", default="default")
        await update.message.reply_text("🗜 Compacting conversation…")  # type: ignore[union-attr]
        try:
            ran = await self._execute_compact(_thread_id(chat_id), agent_version)
            await update.message.reply_text("✅ Done." if ran else "Nothing to compact.")  # type: ignore[union-attr]
        except ValueError as e:
            await update.message.reply_text(str(e))  # type: ignore[union-attr]
        except Exception:
            logger.exception("Manual compaction failed")
            await update.message.reply_text("Compaction failed — check logs.")  # type: ignore[union-attr]

    @_restricted
    async def _handle_consolidate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Running memory consolidation...")  # type: ignore[union-attr]
        try:
            ran = await run_light_consolidation()
            msg = "Done." if ran else "Nothing to consolidate — no notes."
            await update.message.reply_text(msg)  # type: ignore[union-attr]
        except Exception:
            logger.exception("Manual consolidation failed")
            await update.message.reply_text("Consolidation failed — check logs.")  # type: ignore[union-attr]

    @_restricted
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

    @_restricted
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

    @_restricted
    async def _handle_approvals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List saved approval rules with per-rule Revoke buttons."""
        rules = list_approvals()
        if not rules:
            await update.message.reply_text("No saved approval rules.")  # type: ignore[union-attr]
            return
        lines = []
        buttons = []
        for i, rule in enumerate(rules, start=1):
            tool = rule.get("tool", "*")
            target = rule.get("target", "*")
            pattern = rule.get("pattern", "?")
            lines.append(
                f"{i}. <code>{_escape(tool)}</code> @ <code>{_escape(target)}</code>"
                f": <code>{_escape(pattern)}</code>"
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

    @_restricted
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

    @_restricted
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


def _format_tool_call(
    tool_name: str, args: dict, done: bool, spin: int = 0, error: bool = False
) -> str:
    icon = ("❌" if error else "🟢") if done else _SPINNER[spin % len(_SPINNER)]
    display = _TOOL_NAMES.get(tool_name, tool_name)
    if tool_name == "fetch_page":
        urls = args.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        links = ", ".join(
            f'<a href="{_escape(url)}">{_escape(urlparse(url).netloc or url)}</a>' for url in urls
        )
        return f"{icon} <code>{_escape(display)}(</code>{links}<code>)</code>"
    if tool_name == "respond_with_file":
        filename = _escape(args.get("filename", "file"))
        return f"{icon} <code>{_escape(display)}({filename})</code>"
    inner = _format_args(args)
    call = f"{display}({inner})" if inner else f"{display}()"
    return f"{icon} <code>{_escape(call)}</code>"


def _format_args(args: dict) -> str:
    if not args:
        return ""
    value = next(iter(args.values()))
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value)
    if len(text) > _ARG_TRUNCATE:
        text = text[:_ARG_TRUNCATE] + "…"
    return text


async def _spinner_task(msg, tool_name: str, args: dict, step_holder: list[str]) -> None:
    i = 0
    while True:
        await asyncio.sleep(1.0)
        i += 1
        try:
            header = _format_tool_call(tool_name, args, done=False, spin=i)
            step = step_holder[0]
            text = f"{header}\n<code>{_escape(step)}</code>" if step else header
            await msg.edit_text(text, parse_mode="HTML", link_preview_options=_NO_PREVIEW)
        except Exception:
            return


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
    while not stop_event.is_set():
        await update.get_bot().send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
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
        self._out.append(_escape(data))

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
