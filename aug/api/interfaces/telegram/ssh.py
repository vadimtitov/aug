"""Telegram handlers for SSH target management (/ssh command).

Provides _SshMixin, which TelegramInterface inherits. All SSH-related
conversation states, handlers, and handler registration live here.

Flow:
  /ssh → menu (Add / List / Remove)
  Add  → name → host → port → user → password (one-time) →
         provision (generate Ed25519 key, install, capture fingerprint) →
         fingerprint confirm → save target (password never stored)
  List → inline list of configured targets
  Remove → inline list → tap to remove
"""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from aug.api.interfaces.telegram.utils import _escape, _restricted
from aug.utils.ssh import (
    cleanup_keys,
    find_target,
    get_targets,
    provision_target,
    remove_target,
    save_target,
)

logger = logging.getLogger(__name__)

(
    _SSH_ADD_NAME,
    _SSH_ADD_HOST,
    _SSH_ADD_PORT,
    _SSH_ADD_USER,
    _SSH_ADD_PASSWORD,
    _SSH_CONFIRM_FP,
) = range(2, 8)


class _SshMixin:
    """Mixin that adds /ssh target management to TelegramInterface."""

    def build_ssh_handlers(self, bot_app) -> None:
        """Register all SSH-related handlers on *bot_app*."""
        bot_app.add_handler(CommandHandler("ssh", self._handle_ssh))
        bot_app.add_handler(
            ConversationHandler(
                entry_points=[CallbackQueryHandler(self._ssh_add_start, pattern=r"^ssh:add$")],
                states={
                    _SSH_ADD_NAME: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._ssh_got_name)
                    ],
                    _SSH_ADD_HOST: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._ssh_got_host)
                    ],
                    _SSH_ADD_PORT: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._ssh_got_port)
                    ],
                    _SSH_ADD_USER: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._ssh_got_user)
                    ],
                    _SSH_ADD_PASSWORD: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self._ssh_got_password)
                    ],
                    _SSH_CONFIRM_FP: [
                        CallbackQueryHandler(self._ssh_confirm_fp, pattern=r"^ssh_fp:(yes|no)$")
                    ],
                    ConversationHandler.TIMEOUT: [TypeHandler(object, self._ssh_timeout)],
                },
                fallbacks=[CommandHandler("cancel", self._ssh_cancel)],
                conversation_timeout=300,
            )
        )
        bot_app.add_handler(CallbackQueryHandler(self._ssh_list, pattern=r"^ssh:list$"))
        bot_app.add_handler(CallbackQueryHandler(self._ssh_remove_menu, pattern=r"^ssh:remove$"))
        bot_app.add_handler(CallbackQueryHandler(self._ssh_remove_target, pattern=r"^ssh:remove:"))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    @_restricted
    async def _handle_ssh(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show the /ssh management menu."""
        buttons = [
            [InlineKeyboardButton("+ Add target", callback_data="ssh:add")],
            [InlineKeyboardButton("List targets", callback_data="ssh:list")],
            [InlineKeyboardButton("Remove target", callback_data="ssh:remove")],
        ]
        await update.message.reply_text(  # type: ignore[union-attr]
            "SSH target management:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    @_restricted
    async def _ssh_add_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Entry point for the Add conversation — triggered by the Add button."""
        query = update.callback_query
        if query is None:
            return ConversationHandler.END
        await query.answer()
        await query.edit_message_text("Enter a name for this target (e.g. homeserver):")
        return _SSH_ADD_NAME

    async def _ssh_got_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        name = (update.message.text or "").strip()  # type: ignore[union-attr]
        if not name:
            await update.message.reply_text("Name cannot be empty. Try again:")  # type: ignore[union-attr]
            return _SSH_ADD_NAME
        if find_target(name) is not None:
            await update.message.reply_text(  # type: ignore[union-attr]
                f"A target named <code>{_escape(name)}</code> already exists. "
                "Remove it first via /ssh.",
                parse_mode="HTML",
            )
            return ConversationHandler.END
        context.user_data["ssh_add"] = {"name": name}
        await update.message.reply_text("Enter host (IP or hostname):")  # type: ignore[union-attr]
        return _SSH_ADD_HOST

    async def _ssh_got_host(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        host = (update.message.text or "").strip()  # type: ignore[union-attr]
        if not host:
            await update.message.reply_text("Host cannot be empty. Try again:")  # type: ignore[union-attr]
            return _SSH_ADD_HOST
        context.user_data["ssh_add"]["host"] = host
        await update.message.reply_text("Enter port (default: 22):")  # type: ignore[union-attr]
        return _SSH_ADD_PORT

    async def _ssh_got_port(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        raw = (update.message.text or "").strip()  # type: ignore[union-attr]
        try:
            port = int(raw) if raw else 22
        except ValueError:
            await update.message.reply_text("Port must be a number. Try again:")  # type: ignore[union-attr]
            return _SSH_ADD_PORT
        context.user_data["ssh_add"]["port"] = port
        await update.message.reply_text("Enter username:")  # type: ignore[union-attr]
        return _SSH_ADD_USER

    async def _ssh_got_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = (update.message.text or "").strip()  # type: ignore[union-attr]
        if not user:
            await update.message.reply_text("Username cannot be empty. Try again:")  # type: ignore[union-attr]
            return _SSH_ADD_USER
        context.user_data["ssh_add"]["user"] = user
        context.user_data["ssh_add"]["prompt_msg_ids"] = [update.message.message_id]
        msg = await update.message.reply_text(  # type: ignore[union-attr]
            "Enter password (used once to install the SSH key, never stored):"
        )
        context.user_data["ssh_add"]["prompt_msg_ids"].append(msg.message_id)
        return _SSH_ADD_PASSWORD

    async def _ssh_got_password(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        password = update.message.text or ""  # type: ignore[union-attr]
        password_msg_id = update.message.message_id  # type: ignore[union-attr]

        # Delete password message and prompt immediately for privacy
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        for msg_id in [
            *context.user_data["ssh_add"].get("prompt_msg_ids", []),
            password_msg_id,
        ]:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception:
                pass

        data = context.user_data["ssh_add"]
        status_msg = await update.effective_chat.send_message(  # type: ignore[union-attr]
            f"⏳ Provisioning <code>{_escape(data['name'])}</code>…", parse_mode="HTML"
        )

        try:
            key_path, known_hosts_path, fingerprint = await provision_target(
                name=data["name"],
                host=data["host"],
                port=data["port"],
                user=data["user"],
                password=password,
            )
        except Exception as exc:
            logger.warning("ssh_provision_failed name=%s error=%r", data["name"], exc)
            cleanup_keys(data["name"])
            context.user_data.pop("ssh_add", None)
            await status_msg.edit_text(
                f"❌ Provisioning failed: <code>{_escape(str(exc))}</code>",
                parse_mode="HTML",
            )
            return ConversationHandler.END

        context.user_data["ssh_add"].update(
            {
                "key_path": key_path,
                "known_hosts_path": known_hosts_path,
                "fingerprint": fingerprint,
            }
        )

        buttons = [
            [
                InlineKeyboardButton("Yes, save it", callback_data="ssh_fp:yes"),
                InlineKeyboardButton("No, abort", callback_data="ssh_fp:no"),
            ]
        ]
        await status_msg.edit_text(
            f"🔑 <b>Host fingerprint for {_escape(data['host'])}:</b>\n"
            f"<code>{_escape(fingerprint)}</code>\n\n"
            "Is this your server?",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return _SSH_CONFIRM_FP

    async def _ssh_confirm_fp(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        if query is None:
            return ConversationHandler.END
        await query.answer()

        confirmed = (query.data or "").split(":")[1] == "yes"
        data = context.user_data.get("ssh_add", {})

        if confirmed:
            save_target(
                name=data["name"],
                host=data["host"],
                port=data["port"],
                user=data["user"],
                key_path=data["key_path"],
                known_hosts_path=data["known_hosts_path"],
            )
            await query.edit_message_text(
                f"✅ Target <code>{_escape(data['name'])}</code> saved.",
                parse_mode="HTML",
            )
        else:
            cleanup_keys(data["name"])
            await query.edit_message_text("❌ Provisioning cancelled.")

        context.user_data.pop("ssh_add", None)
        return ConversationHandler.END

    async def _ssh_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.pop("ssh_add", None)
        await update.message.reply_text("Cancelled.")  # type: ignore[union-attr]
        return ConversationHandler.END

    async def _ssh_timeout(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Clean up stale user_data when an SSH add conversation times out."""
        name = (context.user_data.pop("ssh_add", None) or {}).get("name")
        if name:
            cleanup_keys(name)
        return ConversationHandler.END

    @_restricted
    async def _ssh_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        targets = get_targets()
        if not targets:
            await query.edit_message_text("No SSH targets configured.")
            return
        lines = [
            f"• <code>{_escape(t.get('name', '?'))}</code> — "
            f"{_escape(t.get('user', '?'))}@{_escape(t.get('host', '?'))}:{t.get('port', 22)}"
            for t in targets
        ]
        await query.edit_message_text(
            "Configured SSH targets:\n\n" + "\n".join(lines),
            parse_mode="HTML",
        )

    @_restricted
    async def _ssh_remove_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        targets = get_targets()
        if not targets:
            await query.edit_message_text("No SSH targets configured.")
            return
        buttons = [
            [InlineKeyboardButton(t.get("name", "?"), callback_data=f"ssh:remove:{t.get('name')}")]
            for t in targets
        ]
        await query.edit_message_text(
            "Select a target to remove:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    @_restricted
    async def _ssh_remove_target(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        parts = (query.data or "").split(":", 2)
        if len(parts) != 3:
            return
        name = parts[2]
        remove_target(name)
        cleanup_keys(name)
        await query.edit_message_text(
            f"Target <code>{_escape(name)}</code> removed.",
            parse_mode="HTML",
        )
