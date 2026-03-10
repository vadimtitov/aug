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

import markdown as md
from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from aug.config import get_settings
from aug.core.registry import get_agent
from aug.core.state import AgentState

logger = logging.getLogger(__name__)


# Tags supported by Telegram's HTML parse mode.
_TELEGRAM_TAGS = {"b", "strong", "i", "em", "code", "pre", "a", "s", "u", "span"}


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

    Converts to HTML then strips any tags Telegram doesn't support,
    keeping their inner content. Tables are rendered as monospaced <pre> blocks.
    """
    html = md.markdown(text, extensions=["fenced_code", "tables"])
    # Convert tables to <pre> blocks before stripping.
    html = re.sub(r"<table[^>]*>.*?</table>", _table_to_pre, html, flags=re.DOTALL)
    # Replace unsupported block tags with newlines to preserve structure.
    html = re.sub(r"<(p|li|tr|th|td|h[1-6]|blockquote)([^>]*)>", "\n", html)
    # Strip all remaining unsupported tags, keeping content.
    html = re.sub(r"</?(?!(?:" + "|".join(_TELEGRAM_TAGS) + r")\b)[a-zA-Z][^>]*>", "", html)
    # Collapse excessive blank lines.
    html = re.sub(r"\n{3,}", "\n\n", html).strip()
    return html


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


def _thread_id(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    session = context.application.bot_data.get(f"session:{chat_id}", 0)
    return f"tg-{chat_id}-{session}"


async def _handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new conversation thread, discarding the current context."""
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    current = context.application.bot_data.get(f"session:{chat_id}", 0)
    context.application.bot_data[f"session:{chat_id}"] = current + 1
    await update.message.reply_text("Context cleared. Starting fresh.")  # type: ignore[union-attr]


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a Telegram message through the default agent."""
    if not update.message or not update.message.text:
        return

    thread_id = _thread_id(context, update.effective_chat.id)  # type: ignore[union-attr]
    checkpointer = context.application.bot_data["checkpointer"]
    graph = get_agent("default", checkpointer)
    input_state = AgentState(
        messages=[HumanMessage(content=update.message.text)],
        thread_id=thread_id,
        interface_context=(
            "Interface: Telegram.\n"
            "Formatting: bold and italic are supported. "
            "Do not use markdown tables — present tabular data as bullet lists instead. "
            "Keep responses concise; this is a mobile messaging app."
        ),
    )

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(update, stop_typing))

    accumulated_text = ""
    status_msg = None

    try:
        config = {"configurable": {"thread_id": thread_id}}
        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                delta = event["data"]["chunk"].content
                if delta:
                    accumulated_text += delta
            elif kind == "on_tool_start":
                status_msg = await update.message.reply_text(f"🔧 {event['name']}...")
            elif kind == "on_tool_end":
                if status_msg:
                    await status_msg.edit_text(f"✅ {event['name']} done")
                accumulated_text = ""

        text = accumulated_text or "..."
        html = _to_html(text)
        try:
            await update.message.reply_text(html, parse_mode="HTML")
        except Exception:
            logger.warning("HTML parse failed, sending as plain text. html=%r", html[:500])
            await update.message.reply_text(text)

    except Exception as e:
        logger.exception("Error handling Telegram message")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")

    finally:
        stop_typing.set()
        typing_task.cancel()


def build_bot(checkpointer) -> Application:
    bot_app = Application.builder().token(get_settings().TELEGRAM_BOT_TOKEN).build()  # type: ignore[arg-type]
    bot_app.bot_data["checkpointer"] = checkpointer
    bot_app.add_handler(CommandHandler("clear", _handle_clear))
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
