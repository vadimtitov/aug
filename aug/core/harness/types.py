"""Harness-agnostic types: what goes into a turn and what comes out of it.

A harness (LangGraph today, Claude Code later) takes a Message for a context_id and
streams events back.  Interfaces build the Message and render the events; nothing in
here knows which harness runs the turn.

Events that belong to a tool call carry ``parent_id``: the id of the ToolStarted they
belong under.  A subagent's own tool calls are ToolStarted events whose parent is the
subagent's call.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from aug.core.tools.output import ToolOutput


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
    follow_up: bool = False  # a turn for messages the previous turn left unconsumed


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
class ApprovalRequest:
    """An operation a tool wants the user to approve before it runs.

    tool_name: the tool asking (e.g. ``"run_ssh"``).
    resource:  what is acted on (an SSH target, a file path); "" if the tool has none.
    operation: what is done (a shell command, a file transfer).
    """

    tool_name: str
    resource: str
    operation: str

    @property
    def description(self) -> str:
        """Human-readable string used in denial messages and fallback display."""
        return f"{self.resource}: {self.operation}" if self.resource else self.operation


class ApprovalDecision(Enum):
    APPROVED_ONCE = "approved_once"
    APPROVED_ALWAYS = "approved_always"
    DENIED = "denied"
    # Never chosen by the user; each tells the model why the tool did not run.
    REPLIED = "replied"  # the user typed a message instead of tapping a button
    TIMED_OUT = "timed_out"
    NO_USER = "no_user"  # an unattended turn: nobody is there to ask

    @property
    def approved(self) -> bool:
        return self in (ApprovalDecision.APPROVED_ONCE, ApprovalDecision.APPROVED_ALWAYS)


@dataclass(frozen=True)
class TextDelta:
    text: str
    parent_id: str | None = None


@dataclass(frozen=True)
class ToolStarted:
    id: str
    name: str  # canonical: no harness prefix, so display and approval rules match
    args: dict
    parent_id: str | None = None


@dataclass(frozen=True)
class ToolProgress:
    """A step a running tool reports (a browser step, say)."""

    text: str
    parent_id: str | None = None


@dataclass(frozen=True)
class ToolEnded:
    id: str
    name: str
    output: ToolOutput | str | None
    error: bool = False  # also True when the tool's approval was not granted
    parent_id: str | None = None


@dataclass(frozen=True)
class StatusUpdate:
    """A brief notice outside any tool (e.g. 'Compacting conversation…')."""

    text: str
    parent_id: str | None = None


@dataclass(frozen=True)
class ApprovalRequested:
    """A tool is waiting for the user's decision on *request*.

    The interface asks the user and answers with ``pending_approvals.resolve(id, …)``.
    """

    id: str
    request: ApprovalRequest
    parent_id: str | None = None


@dataclass(frozen=True)
class ApprovalResolved:
    id: str
    decision: ApprovalDecision
    parent_id: str | None = None


AgentEvent = (
    TextDelta
    | ToolStarted
    | ToolProgress
    | ToolEnded
    | StatusUpdate
    | ApprovalRequested
    | ApprovalResolved
)


@dataclass(frozen=True)
class TurnStarted:
    context_id: str


@dataclass(frozen=True)
class TurnEnded:
    """Always the last event of a turn.

    unconsumed: messages injected too late for the turn to use; the caller starts a
                follow-up turn with them.
    """

    unconsumed: list[Message] = field(default_factory=list)
    error: HarnessError | None = None
    cancelled: bool = False
    usage: Usage = field(default_factory=Usage)


HarnessEvent = AgentEvent | TurnStarted | TurnEnded

# Receives a turn the harness started by itself (Claude Code finishing background work).
UnsolicitedSink = Callable[[str, AsyncIterator[HarnessEvent]], Awaitable[None]]
