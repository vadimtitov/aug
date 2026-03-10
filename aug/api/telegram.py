"""Telegram bot — polling mode.

Telegram is treated as just another frontend. Incoming messages are routed
through the same LangGraph graph as /chat/invoke.

Polling runs as a background task started during FastAPI lifespan — no public
URL or webhook registration needed.

If TELEGRAM_BOT_TOKEN is not set the bot is silently disabled.
"""

import logging

from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from aug.config import settings
from aug.core.graph import get_agent

logger = logging.getLogger(__name__)


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a Telegram message through the default agent and reply."""
    if not update.message or not update.message.text:
        return

    # Each Telegram chat gets its own persistent thread.
    thread_id = f"tg-{update.effective_chat.id}"  # type: ignore[union-attr]

    checkpointer = context.application.bot_data["checkpointer"]
    graph = get_agent("default", checkpointer)

    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=update.message.text)], "thread_id": thread_id},
        config=config,
    )

    last_ai = next((m for m in reversed(result["messages"]) if m.type == "ai"), None)
    await update.message.reply_text(last_ai.content if last_ai else "Something went wrong.")


def build_bot(checkpointer) -> Application:
    """Build the bot Application with handlers registered."""
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
