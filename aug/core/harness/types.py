"""Harness-agnostic types: what goes into a turn and what comes out of it.

A harness (LangGraph today, Claude Code later) takes a Message for a context_id and
streams events back.  Interfaces build the Message and render the events; nothing in
here knows which harness runs the turn.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from aug.core.events import AgentEvent as GraphEvent
from aug.core.tools.approval import ApprovalRequest


@dataclass(frozen=True)
class Sender:
    """Who sent a message, and through which delivery surface."""

    interface: str  # "telegram", "rest_api": also picks the interface text in the prompt
    user_id: str  # the person; "" when nobody sent it (pushes, jobs)


@dataclass(frozen=True)
class Address:
    """Where a turn's replies, files and scheduled follow-ups go."""

    interface: str
    conversation_id: str  # stable chat id: survives /clear
    chat_id: str = ""  # the interface's own delivery id (Telegram chat id, REST thread)


@dataclass(frozen=True)
class TurnContext:
    reply_to: Address
    attended: bool  # False for pushes and jobs: nobody is there to answer an approval


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class FilePart:
    path: Path  # already saved to disk by the interface; never bytes
    media_type: str
    filename: str


Part = TextPart | FilePart


@dataclass(frozen=True)
class Message:
    parts: list[Part]
    sender: Sender

    @property
    def text(self) -> str:
        """The text parts joined, for logs and the transcript."""
        return "\n\n".join(p.text for p in self.parts if isinstance(p, TextPart))


class ModelTier(Enum):
    STRONG = "strong"
    FAST = "fast"


ErrorKind = Literal["context_full", "rate_limited", "auth", "unavailable", "step_limit", "internal"]


@dataclass(frozen=True)
class HarnessError:
    kind: ErrorKind
    retry_at: datetime | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class TurnStarted:
    context_id: str


@dataclass(frozen=True)
class TurnEnded:
    """Always the last event of a turn.

    unconsumed: messages injected too late for the turn to use; the caller starts a
                follow-up turn with them.
    approval:   the turn paused on this approval request (LangGraph's interrupt).  The
                caller asks the user and resumes.  Goes away once approvals are awaited
                inside the tool (ToolContext.request_approval).
    """

    unconsumed: list[Message] = field(default_factory=list)
    error: HarnessError | None = None
    cancelled: bool = False
    usage: Usage = field(default_factory=Usage)
    approval: ApprovalRequest | None = None


HarnessEvent = GraphEvent | TurnStarted | TurnEnded

# Receives a turn the harness started by itself (Claude Code finishing background work).
UnsolicitedSink = Callable[[str, AsyncIterator[HarnessEvent]], Awaitable[None]]
