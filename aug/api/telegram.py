"""Telegram bot — polling mode.

Telegram is treated as just another frontend. Incoming messages are routed
through the same LangGraph graph as /chat/invoke.

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed.

If TELEGRAM_BOT_TOKEN is not set the bot is silently disabled.

Message UX:
- "typing..." chat action is sent (and refreshed) while the agent works.
- A placeholder message is sent immediately, then edited as events arrive:
    • tool_call  → "🔧 calling <tool>..."
    • tool_result → "✅ <tool> done"
    • streaming tokens → accumulated and flushed every ~1s (Telegram rate limit)
- Final edit is the complete response.
"""

import asyncio
import logging

from langchain_core.messages import HumanMessage
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from aug.config import settings
from aug.core.graph import get_agent

logger = logging.getLogger(__name__)

# How often (seconds) to flush accumulated token deltas to Telegram.
# Telegram allows ~1 edit/second per message before rate-limiting.
_EDIT_INTERVAL = 1.0


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


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a Telegram message through the default agent with live status updates."""
    if not update.message or not update.message.text:
        return

    thread_id = f"tg-{update.effective_chat.id}"  # type: ignore[union-attr]
    checkpointer = context.application.bot_data["checkpointer"]
    graph = get_agent("default", checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=update.message.text)],
        "thread_id": thread_id,
    }

    # Start typing indicator in background.
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(update, stop_typing))

    # Send a placeholder message we'll edit as events arrive.
    status_msg: Message = await update.message.reply_text("...")

    accumulated_text = ""
    last_edit_text = ""
    last_edit_time = 0.0

    async def _flush(text: str, force: bool = False) -> None:
        """Edit the status message if text changed and enough time has passed."""
        nonlocal last_edit_text, last_edit_time
        now = asyncio.get_event_loop().time()
        if text == last_edit_text:
            return
        if not force and (now - last_edit_time) < _EDIT_INTERVAL:
            return
        try:
            await status_msg.edit_text(text)
            last_edit_text = text
            last_edit_time = now
        except Exception:
            pass  # ignore "message not modified" errors from Telegram

    try:
        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                delta = event["data"]["chunk"].content
                if delta:
                    accumulated_text += delta
                    await _flush(accumulated_text)

            elif kind == "on_tool_start":
                await _flush(f"🔧 calling {event['name']}...", force=True)

            elif kind == "on_tool_end":
                await _flush(f"✅ {event['name']} done", force=True)
                accumulated_text = ""  # reset so next LLM turn starts fresh

        # Final edit with complete response.
        await _flush(accumulated_text or "...", force=True)

    except Exception as e:
        logger.exception("Error handling Telegram message")
        await status_msg.edit_text(f"Sorry, something went wrong: {e}")

    finally:
        stop_typing.set()
        typing_task.cancel()


def build_bot(checkpointer) -> Application:
    bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()  # type: ignore[arg-type]
    bot_app.bot_data["checkpointer"] = checkpointer
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    return bot_app


async def start_polling(app) -> None:
    """Start the bot and begin polling. Called from FastAPI lifespan."""
    if not settings.TELEGRAM_BOT_TOKEN:
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
