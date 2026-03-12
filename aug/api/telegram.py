"""Telegram bot — polling mode.

Telegram is treated as just another frontend. Incoming messages are routed
through the same LangGraph graph as /chat/invoke.

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed.

If TELEGRAM_BOT_TOKEN is not set the bot is silently disabled.

Message UX:
- "typing..." chat action is sent (and refreshed) while the agent works.
- The final response is sent as a single message once the agent finishes.
"""

import asyncio
import logging
import re
import subprocess
import textwrap
from html.parser import HTMLParser
from urllib.parse import urlparse

import markdown as md
from langchain_core.messages import HumanMessage
from openai import RateLimitError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ChatAction
from telegram.error import RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from aug.config import get_settings
from aug.core.registry import get_agent, list_agents
from aug.core.state import AgentState
from aug.utils.user_settings import get_setting, set_setting

logger = logging.getLogger(__name__)


_SPINNER = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]

_TOOL_NAMES = {
    "brave_search": "Search",
    "fetch_page": "Fetch",
    "run_bash": "Bash",
    "remember": "Remember",
    "recall": "Recall",
    "update_memory": "Memory",
    "forget": "Forget",
}
_ARG_TRUNCATE = 50
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)


def build_bot(checkpointer) -> Application:
    bot_app = Application.builder().token(get_settings().TELEGRAM_BOT_TOKEN).build()  # type: ignore[arg-type]
    bot_app.bot_data["checkpointer"] = checkpointer
    bot_app.add_handler(
        ConversationHandler(
            entry_points=[CommandHandler("secret", _secret_start)],
            states={
                _SECRET_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, _secret_got_name)],
                _SECRET_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, _secret_got_value)],
            },
            fallbacks=[],
        )
    )
    bot_app.add_handler(CommandHandler("clear", _handle_clear))
    bot_app.add_handler(CommandHandler("version", _handle_version))
    bot_app.add_handler(CallbackQueryHandler(_handle_version_callback, pattern=r"^version:"))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    return bot_app


async def start_polling(app) -> None:
    """Start the bot and begin polling. Called from FastAPI lifespan."""
    if not get_settings().TELEGRAM_BOT_TOKEN:
        logger.info("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
        return

    bot_app = build_bot(app.state.checkpointer)
    app.state.telegram = bot_app

    await bot_app.initialize()
    await bot_app.bot.set_my_commands(
        [
            ("version", "Switch agent version"),
            ("secret", "Store a secret"),
            ("clear", "Start a new conversation"),
        ]
    )
    await bot_app.start()
    await bot_app.updater.start_polling()
    logger.info("Telegram bot started (polling).")


async def stop_polling(app) -> None:
    """Gracefully stop the bot. Called from FastAPI lifespan shutdown."""
    bot_app = getattr(app.state, "telegram", None)
    if bot_app is None:
        return
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()
    logger.info("Telegram bot stopped.")


def _format_tool_call(tool_name: str, args: dict, done: bool, spin: int = 0) -> str:
    icon = "✅" if done else _SPINNER[spin % len(_SPINNER)]
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


_TELEGRAM_TAGS = {"b", "strong", "i", "em", "code", "pre", "a", "s", "u", "span", "blockquote"}
_BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th"}


class _TelegramSanitizer(HTMLParser):
    """Strip HTML down to only Telegram-supported tags.

    Keeps content of unsupported tags. Ignores orphaned closing tags.
    Ensures all opened allowed tags are closed.
    """

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
    """Convert an HTML table to a monospaced <pre> block."""
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
    """Convert markdown to Telegram-compatible HTML.

    Converts to HTML then sanitizes down to only Telegram-supported tags.
    Tables are rendered as monospaced <pre> blocks.
    """
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    html = re.sub(r"<table[^>]*>.*?</table>", _table_to_pre, html, flags=re.DOTALL)
    sanitizer = _TelegramSanitizer()
    sanitizer.feed(html)
    return sanitizer.result()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _typing_loop(update: Update, stop_event: asyncio.Event) -> None:
    """Send 'typing' chat action every 4s until stop_event is set.

    Telegram's typing indicator times out after 5s, so we refresh at 4s.
    """
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    while not stop_event.is_set():
        await update.get_bot().send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4.0)
        except TimeoutError:
            pass


def _is_allowed(chat_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or chat_id in allowed


def _thread_id(chat_id: int) -> str:
    session = get_setting("telegram", "chats", str(chat_id), "session", default=0)
    return f"tg-{chat_id}-{session}"


_SECRET_NAME, _SECRET_VALUE = range(2)


async def _secret_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not _is_allowed(update.effective_chat.id):  # type: ignore[union-attr]
        return ConversationHandler.END
    msg = await update.message.reply_text("Enter secret name:")  # type: ignore[union-attr]
    context.user_data["secret_msgs"] = [update.message.message_id, msg.message_id]  # type: ignore[union-attr]
    return _SECRET_NAME


async def _secret_got_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["secret_name"] = update.message.text  # type: ignore[union-attr]
    context.user_data["secret_msgs"].append(update.message.message_id)  # type: ignore[union-attr]
    msg = await update.message.reply_text("Enter secret value:")  # type: ignore[union-attr]
    context.user_data["secret_msgs"].append(msg.message_id)
    return _SECRET_VALUE


async def _secret_got_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["secret_msgs"].append(update.message.message_id)  # type: ignore[union-attr]
    name = context.user_data["secret_name"]
    value = update.message.text  # type: ignore[union-attr]

    result = subprocess.run(
        ["hushed", "add", name, value],
        capture_output=True,
        text=True,
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


async def _handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new conversation thread, discarding the current context."""
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    if not _is_allowed(chat_id):
        return
    current = get_setting("telegram", "chats", str(chat_id), "session", default=0)
    set_setting("telegram", "chats", str(chat_id), "session", value=current + 1)
    await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]


async def _handle_version(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available agent versions as inline buttons."""
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    if not _is_allowed(chat_id):
        return
    current = get_setting("telegram", "chats", str(chat_id), "agent", default="default")
    agents = [a for a in list_agents() if a != "fake"]
    buttons = [
        [InlineKeyboardButton(f"{'✅ ' if a == current else ''}{a}", callback_data=f"version:{a}")]
        for a in agents
    ]
    await update.message.reply_text(  # type: ignore[union-attr]
        f"Current version: <code>{_escape(current)}</code>\nChoose a version:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def _handle_version_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle version selection from inline keyboard."""
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


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a Telegram message through the default agent."""
    if not update.message or not update.message.text:
        return
    if not _is_allowed(update.effective_chat.id):  # type: ignore[union-attr]
        return

    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    thread_id = _thread_id(chat_id)
    checkpointer = context.application.bot_data["checkpointer"]
    agent_name = get_setting("telegram", "chats", str(chat_id), "agent", default="default")
    graph = get_agent(agent_name, checkpointer)
    input_state = AgentState(
        messages=[HumanMessage(content=update.message.text)],
        thread_id=thread_id,
        interface_context="Telegram bot. Keep responses concise — there is a message length limit.",
        response_format=textwrap.dedent("""
            Use HTML formatting only. Do NOT use Markdown syntax — no *asterisks*,
            no _underscores_, no **double asterisks**, no # headers, no --- dividers.
            It will appear as raw symbols to the user.

            Supported tags:
            - <b>bold</b>
            - <i>italic</i>
            - <u>underline</u>
            - <s>strikethrough</s>
            - <code>inline code</code>
            - <pre>code block</pre>
            - <a href="URL">link text</a>
            - <span class="tg-spoiler">spoiler</span>
            - <blockquote>quote</blockquote>

            In plain text, escape: & → &amp;  < → &lt;  > → &gt;

            Tables are not supported. Use labeled lists instead:

            WRONG:
            | Name  | Price |
            |-------|-------|
            | Apple | £1.00 |
            | Pear  | £0.80 |

            CORRECT:
            <b>Apple</b> — £1.00
            <b>Pear</b> — £0.80
        """).strip(),
    )
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(update, stop_typing))

    accumulated_text = ""
    stream_msg = None
    last_stream_edit = 0.0
    tool_msgs: dict[str, tuple] = {}

    try:
        config = {"configurable": {"thread_id": thread_id}}
        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                delta = event["data"]["chunk"].content
                if delta:
                    accumulated_text += delta
                    now = asyncio.get_running_loop().time()
                    if stream_msg is None:
                        stream_msg = await update.message.reply_text(
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
                msg = await update.message.reply_text(
                    text, parse_mode="HTML", link_preview_options=_NO_PREVIEW
                )
                spin = asyncio.create_task(_spinner_task(msg, event["name"], args))
                tool_msgs[event["run_id"]] = (args, msg, spin)
                accumulated_text = ""
                stream_msg = None
            elif kind == "on_tool_end":
                entry = tool_msgs.pop(event["run_id"], None)
                if entry:
                    args, msg, spin = entry
                    spin.cancel()
                    try:
                        await msg.edit_text(
                            _format_tool_call(event["name"], args, done=True),
                            parse_mode="HTML",
                            link_preview_options=_NO_PREVIEW,
                        )
                    except Exception:
                        pass

        if accumulated_text:
            final_text, final_parse_mode = _to_html(accumulated_text), "HTML"
            try:
                if stream_msg is not None:
                    await stream_msg.edit_text(
                        final_text,
                        parse_mode=final_parse_mode,
                        link_preview_options=_NO_PREVIEW,
                    )
                else:
                    await update.message.reply_text(
                        final_text,
                        parse_mode=final_parse_mode,
                        link_preview_options=_NO_PREVIEW,
                    )
            except RetryAfter as e:
                # Flood control hit on final send — wait it out and deliver as a fresh message.
                await asyncio.sleep(e.retry_after)
                await update.message.reply_text(
                    final_text, parse_mode=final_parse_mode, link_preview_options=_NO_PREVIEW
                )
            except Exception:
                logger.warning(
                    "Failed to send with %s, falling back to plain text",
                    final_parse_mode,
                    exc_info=True,
                )
                await update.message.reply_text(accumulated_text, link_preview_options=_NO_PREVIEW)

    except RateLimitError:
        logger.warning("Rate limit / context too large")
        await update.message.reply_text(
            "Context window is full. Use /clear to start a fresh conversation."
        )
    except Exception as e:
        logger.exception("Error handling Telegram message")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")

    finally:
        stop_typing.set()
        typing_task.cancel()
