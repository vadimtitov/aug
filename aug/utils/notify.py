"""Proactive notification dispatch.

send_notification() delivers a message to a user through their preferred interface
without requiring an incoming request context. Used by reminders.
"""

import logging
from typing import Literal

from aug.utils.user_settings import set_setting

logger = logging.getLogger(__name__)


def register_notification_target(thread_id: str, interface: str, sender_id: str) -> None:
    """Persist the sender's interface and ID so reminders can route back to them."""
    set_setting("thread_notifications", thread_id, "interface", value=interface)
    set_setting("thread_notifications", thread_id, "id", value=sender_id)


async def send_notification(app, interface: Literal["telegram"], target_id: str, text: str) -> None:
    """Send a proactive message to a user via their registered interface.

    Raises on failure so callers (e.g. the reminder loop) can retry.

    Args:
        app:       FastAPI application instance (for accessing app.state).
        interface: Interface name, e.g. "telegram".
        target_id: Interface-specific recipient ID, e.g. a Telegram chat_id string.
        text:      Plain text message to deliver.
    """
    if not interface or not target_id:
        raise ValueError(
            f"send_notification: missing target — interface={interface!r} target_id={target_id!r}"
        )

    interfaces = getattr(app.state, "interfaces", {})
    iface = interfaces.get(interface)
    if not iface:
        raise RuntimeError(
            f"send_notification: interface {interface!r} is not registered or not running"
        )

    logger.info("send_notification: sending via %s to %s", interface, target_id)
    await iface.send_notification(target_id, text)
    logger.info("send_notification: delivered via %s to %s", interface, target_id)
