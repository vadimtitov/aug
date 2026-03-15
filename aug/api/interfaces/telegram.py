"""Telegram bot — polling mode.

Telegram is treated as just another frontend via TelegramInterface(BaseInterface[Update]).

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed. If TELEGRAM_BOT_TOKEN is not set the bot
is silently disabled.

Supported input types: text, voice (transcribed), photos, location.
"""

import asyncio
import io
import logging
import re
import subprocess
from collections.abc import AsyncIterator
from html.parser import HTMLParser
from urllib.parse import urlparse

import markdown as md
from langchain_core.runnables.schema import StreamEvent
from langgraph.checkpoint.base import BaseCheckpointSaver
from openai import RateLimitError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from aug.api.interfaces.base import (
    AudioContent,
    BaseInterface,
    ImageContent,
    IncomingMessage,
    LocationContent,
    TextContent,
)
from aug.config import get_settings
from aug.core.prompts import (
    TELEGRAM_INTERFACE_CONTEXT,
    TELEGRAM_RESPONSE_FORMAT,
    build_system_prompt,
)
from aug.core.registry import list_agents
from aug.core.state import AgentState
from aug.core.tools.browser import browser_progress_queue
from aug.utils.user_settings import get_setting, set_setting

logger = logging.getLogger(__name__)

_SPINNER = ["🌑", "🌘", "🌗", "🌖", "🌕", "🌔", "🌓", "🌒"]
_TOOL_NAMES = {
    "brave_search": "Search",
    "fetch_page": "Fetch",
    "run_bash": "Bash",
    "remember": "Remember",
    "recall": "Recall",
    "update_memory": "Memory",
    "forget": "Forget",
    "browser": "Browser",
}
_ARG_TRUNCATE = 50
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
_SECRET_NAME, _SECRET_VALUE = range(2)


class TelegramInterface(BaseInterface[Update]):
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        super().__init__(checkpointer)

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

        parts = []
        if msg.voice:
            file = await msg.voice.get_file()
            parts.append(AudioContent(data=bytes(await file.download_as_bytearray())))
        elif msg.photo:
            file = await msg.photo[-1].get_file()
            parts.append(ImageContent(data=bytes(await file.download_as_bytearray())))
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

        return IncomingMessage(
            parts=parts,
            interface_context=TELEGRAM_INTERFACE_CONTEXT,
            response_format=TELEGRAM_RESPONSE_FORMAT,
            thread_id=_thread_id(chat_id),
            agent_name=get_setting("telegram", "chats", str(chat_id), "agent", default="default"),
        )

    async def send_stream(self, stream: AsyncIterator[StreamEvent], context: Update) -> None:
        msg = context.message
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_typing_loop(context, stop_typing))

        accumulated_text = ""
        stream_msg = None
        last_stream_edit = 0.0
        tool_msgs: dict[str, tuple] = {}

        _browser_queue: asyncio.Queue[str | None] = asyncio.Queue()
        _browser_queue_token = browser_progress_queue.set(_browser_queue)

        try:
            async for event in stream:
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    delta = event["data"]["chunk"].content
                    if delta:
                        accumulated_text += delta
                        now = asyncio.get_running_loop().time()
                        if stream_msg is None:
                            stream_msg = await msg.reply_text(  # type: ignore[union-attr]
                                accumulated_text, link_preview_options=_NO_PREVIEW
                            )
                            last_stream_edit = now
                        elif now - last_stream_edit >= 0.3:
                            try:
                                await stream_msg.edit_text(
                                    accumulated_text, link_preview_options=_NO_PREVIEW
                                )
                                last_stream_edit = now
                            except RetryAfter as e:
                                last_stream_edit = now + e.retry_after
                            except Exception:
                                pass
                elif kind == "on_tool_start":
                    args = event["data"].get("input") or {}
                    text = _format_tool_call(event["name"], args, done=False)
                    tool_msg = await msg.reply_text(  # type: ignore[union-attr]
                        text, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                    )
                    if event["name"] == "browser":
                        consumer = asyncio.create_task(
                            _browser_step_consumer(_browser_queue, tool_msg, event["name"], args)
                        )
                        tool_msgs[event["run_id"]] = (args, tool_msg, consumer)
                    else:
                        spin = asyncio.create_task(_spinner_task(tool_msg, event["name"], args))
                        tool_msgs[event["run_id"]] = (args, tool_msg, spin)
                    accumulated_text = ""
                    stream_msg = None
                elif kind == "on_tool_end":
                    entry = tool_msgs.pop(event["run_id"], None)
                    if entry:
                        args, tool_msg, task = entry
                        task.cancel()
                        try:
                            await tool_msg.edit_text(
                                _format_tool_call(event["name"], args, done=True),
                                parse_mode="HTML",
                                link_preview_options=_NO_PREVIEW,
                            )
                        except Exception:
                            pass

            if accumulated_text:
                final_text = _to_html(accumulated_text)
                try:
                    if stream_msg is not None:
                        await stream_msg.edit_text(
                            final_text, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                        )
                    else:
                        await msg.reply_text(  # type: ignore[union-attr]
                            final_text, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                        )
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after)
                    await msg.reply_text(  # type: ignore[union-attr]
                        final_text, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                    )
                except BadRequest as e:
                    if "not modified" not in str(e).lower():
                        logger.warning("Failed to send HTML, falling back to plain", exc_info=True)
                        await msg.reply_text(accumulated_text, link_preview_options=_NO_PREVIEW)  # type: ignore[union-attr]

        except RateLimitError:
            logger.warning("Rate limit / context too large")
            await msg.reply_text(  # type: ignore[union-attr]
                "Context window is full. Use /clear to start a fresh conversation."
            )
        except Exception as e:
            logger.exception("Error handling Telegram message")
            await msg.reply_text(f"Sorry, something went wrong: {e}")  # type: ignore[union-attr]
        finally:
            browser_progress_queue.reset(_browser_queue_token)
            stop_typing.set()
            typing_task.cancel()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_bot(self) -> Application:
        bot_app = Application.builder().token(get_settings().TELEGRAM_BOT_TOKEN).build()  # type: ignore[arg-type]
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
                },
                fallbacks=[],
            )
        )
        bot_app.add_handler(CommandHandler("clear", self._handle_clear))
        bot_app.add_handler(CommandHandler("version", self._handle_version))
        bot_app.add_handler(CommandHandler("prompt", self._handle_prompt))
        bot_app.add_handler(
            CallbackQueryHandler(self._handle_version_callback, pattern=r"^version:")
        )
        bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        bot_app.add_handler(MessageHandler(filters.VOICE, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.PHOTO, self._handle_input))
        bot_app.add_handler(MessageHandler(filters.LOCATION, self._handle_input))
        return bot_app

    async def start_polling(self, app) -> None:
        """Start the bot and begin polling. Called from FastAPI lifespan."""
        if not get_settings().TELEGRAM_BOT_TOKEN:
            logger.info("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
            return
        bot_app = self.build_bot()
        app.state.telegram = bot_app
        await bot_app.initialize()
        await bot_app.bot.set_my_commands(
            [
                ("version", "Switch agent version"),
                ("secret", "Store a secret"),
                ("clear", "Start a new conversation"),
                ("prompt", "Export current system prompt as a file"),
            ]
        )
        await bot_app.start()
        await bot_app.updater.start_polling()
        logger.info("Telegram bot started (polling).")

    async def stop_polling(self, app) -> None:
        """Gracefully stop the bot. Called from FastAPI lifespan shutdown."""
        bot_app = getattr(app.state, "telegram", None)
        if bot_app is None:
            return
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()
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

    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        if not _is_allowed(chat_id):
            return
        current = get_setting("telegram", "chats", str(chat_id), "session", default=0)
        set_setting("telegram", "chats", str(chat_id), "session", value=current + 1)
        await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]

    async def _handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        if not _is_allowed(chat_id):
            return
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

    async def _handle_version_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        if not _is_allowed(chat_id):
            return
        agent_name = query.data.split(":", 1)[1]  # type: ignore[union-attr]
        if agent_name not in list_agents():
            await query.edit_message_text("Unknown version.")
            return
        set_setting("telegram", "chats", str(chat_id), "agent", value=agent_name)
        await query.edit_message_text(
            f"Switched to <code>{_escape(agent_name)}</code>.", parse_mode="HTML"
        )

    async def _handle_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        if not _is_allowed(chat_id):
            return
        state = AgentState(
            messages=[],
            thread_id=_thread_id(chat_id),
            interface_context=TELEGRAM_INTERFACE_CONTEXT,
            response_format=TELEGRAM_RESPONSE_FORMAT,
        )
        prompt_text = build_system_prompt(state)
        file = io.BytesIO(prompt_text.encode())
        file.name = "system_prompt.txt"
        await update.message.reply_document(document=file, filename="system_prompt.txt")  # type: ignore[union-attr]

    async def _secret_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not _is_allowed(update.effective_chat.id):  # type: ignore[union-attr]
            return ConversationHandler.END
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
        result = subprocess.run(["hushed", "add", name, value], capture_output=True, text=True)
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


# ---------------------------------------------------------------------------
# Module-level helpers called from FastAPI lifespan
# ---------------------------------------------------------------------------


def build_interface(checkpointer: BaseCheckpointSaver) -> TelegramInterface:
    return TelegramInterface(checkpointer)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _is_allowed(chat_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or chat_id in allowed


def _thread_id(chat_id: int) -> str:
    session = get_setting("telegram", "chats", str(chat_id), "session", default=0)
    return f"tg-{chat_id}-{session}"


def _format_tool_call(tool_name: str, args: dict, done: bool, spin: int = 0) -> str:
    icon = "🟢" if done else _SPINNER[spin % len(_SPINNER)]
    display = _TOOL_NAMES.get(tool_name, tool_name)
    if tool_name == "fetch_page":
        urls = args.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        links = ", ".join(
            f'<a href="{_escape(url)}">{_escape(urlparse(url).netloc or url)}</a>' for url in urls
        )
        return f"{icon} <code>{_escape(display)}(</code>{links}<code>)</code>"
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


async def _spinner_task(msg, tool_name: str, args: dict) -> None:
    i = 0
    while True:
        await asyncio.sleep(1.0)
        i += 1
        try:
            await msg.edit_text(
                _format_tool_call(tool_name, args, done=False, spin=i),
                parse_mode="HTML",
                link_preview_options=_NO_PREVIEW,
            )
        except Exception:
            return


async def _browser_step_consumer(queue: asyncio.Queue, msg, tool_name: str, args: dict) -> None:
    step_text = ""
    while True:
        update = await queue.get()
        if update is None:
            break
        step_text = update
        try:
            header = _format_tool_call(tool_name, args, done=False)
            await msg.edit_text(
                f"{header}\n<code>{_escape(step_text)}</code>",
                parse_mode="HTML",
                link_preview_options=_NO_PREVIEW,
            )
        except Exception:
            pass


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
        if tag in _TELEGRAM_TAGS and tag in self._open:
            self._out.append(f"</{tag}>")
            self._open.remove(tag)
        elif tag in _BLOCK_TAGS:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        self._out.append(data)

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
    if not parsed:
        return ""
    widths = [max(len(r[i]) for r in parsed if i < len(r)) for i in range(len(parsed[0]))]
    lines = []
    for i, row in enumerate(parsed):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "<pre>" + "\n".join(lines) + "</pre>"


def _to_html(text: str) -> str:
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    html = re.sub(r"<table[^>]*>.*?</table>", _table_to_pre, html, flags=re.DOTALL)
    sanitizer = _TelegramSanitizer()
    sanitizer.feed(html)
    return sanitizer.result()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
