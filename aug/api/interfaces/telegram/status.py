"""Telegram handlers for the session itself: /status and /compact.

Provides _StatusMixin, which TelegramInterface inherits.  /status shows what the
harness reports about the chat's session (model, context, plan limits, background
work); /compact reports the context before and after.
"""

import logging

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from aug.api.interfaces.telegram.utils import escape, get_context_id, restricted
from aug.core.harness import ContextUsage, LimitUsage, SessionStatus
from aug.core.registry import list_agents

logger = logging.getLogger(__name__)

_BAR_CELLS = 10
# The plan's usage windows by the names Claude gives them.
_LIMIT_NAMES = {
    "five_hour": "5-hour limit",
    "seven_day": "Weekly limit",
    "seven_day_opus": "Weekly Opus limit",
    "seven_day_sonnet": "Weekly Sonnet limit",
}


class _StatusMixin:
    """Mixin that adds /status and /compact to TelegramInterface."""

    def build_status_handlers(self, bot_app) -> None:
        """Register the /status and /compact handlers on *bot_app*."""
        bot_app.add_handler(CommandHandler("status", self._handle_status))
        bot_app.add_handler(CommandHandler("compact", self._handle_compact))

    @restricted
    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if msg is None or update.effective_chat is None:
            return
        context_id = get_context_id(update.effective_chat.id, msg.message_thread_id)
        agent = self.get_agent_version(context_id)  # type: ignore[attr-defined]
        if agent not in list_agents():
            await msg.reply_text("No agent selected. Use /version to pick one first.")
            return
        status = await self._status_or_none(context_id)
        if status is None:
            await msg.reply_text("Couldn't read the session's status — check logs.")
            return
        await msg.reply_text(render_status(agent, status), parse_mode="HTML")

    @restricted
    async def _handle_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if msg is None or update.effective_chat is None:
            return
        context_id = get_context_id(update.effective_chat.id, msg.message_thread_id)
        await msg.reply_text("🗜 Compacting conversation…")
        await msg.reply_text(await self._compact(context_id))

    async def _compact(self, context_id: str) -> str:
        """Compact *context_id*'s session; what to tell the user about it."""
        agent = self.get_agent_version(context_id)  # type: ignore[attr-defined]
        if agent not in list_agents():
            return "No agent selected. Use /version to pick one first."
        before = await self._status_or_none(context_id)
        try:
            ran = await self._execute_compact(context_id, agent)  # type: ignore[attr-defined]
        except ValueError as e:
            return str(e)
        except Exception:
            logger.exception("Manual compaction failed")
            return "Compaction failed — check logs."
        if not ran:
            return "Nothing to compact."
        after = await self._status_or_none(context_id)
        if before is None or after is None or before.context is None or after.context is None:
            return "✅ Done."
        return f"✅ Compacted: {_count(before.context.used)} → {_count(after.context.used)} tokens."

    async def _status_or_none(self, context_id: str) -> SessionStatus | None:
        try:
            return await self.session_status(context_id)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("session status failed context=%s", context_id)
            return None


def render_status(agent: str, status: SessionStatus | None) -> str:
    """*status* as Telegram HTML."""
    if status is None:
        return f"<b>{escape(agent)}</b>"
    head = f"<b>{escape(agent)}</b> · <code>{escape(status.model)}</code>"
    lines = [head, ""]
    lines += _context_lines(status.context)
    if status.limits:
        lines.append("")
        lines += [_limit_line(limit) for limit in status.limits]
    if status.background_tasks:
        n = status.background_tasks
        lines += ["", f"⏳ {n} background task{'s' if n != 1 else ''} running"]
    return "\n".join(lines)


def _context_lines(context: ContextUsage | None) -> list[str]:
    if context is None:
        return ["No context yet: the next message starts a fresh session."]
    approx = "~" if context.estimated else ""
    filled = min(_BAR_CELLS, round(context.percent * _BAR_CELLS / 100))
    bar = "▓" * filled + "░" * (_BAR_CELLS - filled)
    lines = [
        f"<code>{bar}</code> {approx}{context.percent}% · "
        f"{approx}{_count(context.used)} / {_count(context.window)}"
    ]
    if context.compacts_at is not None and context.window:
        lines.append(f"Compacts itself at {round(100 * context.compacts_at / context.window)}%")
    if context.breakdown:
        parts = " · ".join(f"{escape(name)} {approx}{_count(n)}" for name, n in context.breakdown)
        lines.append(f"<i>{parts}</i>")
    return lines


def _limit_line(limit: LimitUsage) -> str:
    name = _LIMIT_NAMES.get(limit.window, limit.window.replace("_", " "))
    line = f"{escape(name)}: {round(limit.utilization * 100)}%"
    if limit.resets_at is not None:
        line += f" · resets {limit.resets_at.strftime('%a %H:%M')} UTC"
    return line


def _count(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    if tokens < 10_000:
        return f"{tokens / 1000:.1f}".removesuffix(".0") + "K"
    if tokens < 1_000_000:
        return f"{tokens // 1000}K"
    return f"{tokens / 1_000_000:.1f}M"
