"""Chat endpoint schemas."""

from datetime import datetime

from pydantic import BaseModel, Field

from aug.core.harness import ContextUsage, LimitUsage, SessionStatus


class Attachment(BaseModel):
    file_id: str = Field(..., description="The file_id POST /files/upload returned.")
    mime_type: str | None = Field(
        default=None, description="Ignored: the type recorded at upload is used."
    )


class ChatRequest(BaseModel):
    context_id: str = Field(
        ..., max_length=128, pattern=r"^[\w\-]+$", description="Conversation thread ID."
    )
    message: str = Field(..., max_length=32_000, description="User message text.")
    agent: str | None = Field(
        default=None,
        max_length=64,
        description="Agent version string. Omit to use the version selected for this thread.",
    )
    attachments: list[Attachment] = Field(default_factory=list)


class ToolCallInfo(BaseModel):
    name: str
    input: dict = Field(default_factory=dict)
    output: dict | None = None


class ChatFile(BaseModel):
    """A file the agent sent; the same fields as the ``file`` SSE event."""

    id: str
    filename: str
    media_type: str
    caption: str = ""
    url: str = Field(..., description="Where to download it: GET, with the API key.")


class ChatResponse(BaseModel):
    context_id: str
    agent: str
    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    files: list[ChatFile] = Field(default_factory=list)


class ApprovalAnswer(BaseModel):
    id: str = Field(..., max_length=64, description="The id from the approval_requested SSE event.")
    decision: str = Field(
        ...,
        pattern=r"^(approved_once|approved_always|denied)$",
        description="Approval decision: approved_once | approved_always | denied",
    )


class PendingApprovalInfo(BaseModel):
    id: str
    tool_name: str
    resource: str
    operation: str


class ContextUsageInfo(BaseModel):
    used: int = Field(..., description="Tokens in the context now.")
    window: int = Field(..., description="The model's context window, in tokens.")
    percent: int
    compacts_at: int | None = Field(
        None, description="Tokens at which the session compacts itself; null if unknown."
    )
    breakdown: dict[str, int] = Field(
        default_factory=dict, description="Tokens per category, when the harness can tell."
    )
    estimated: bool = Field(False, description="True when `used` is an estimate.")

    @classmethod
    def of(cls, usage: ContextUsage) -> "ContextUsageInfo":
        return cls(
            used=usage.used,
            window=usage.window,
            percent=usage.percent,
            compacts_at=usage.compacts_at,
            breakdown=dict(usage.breakdown),
            estimated=usage.estimated,
        )


class LimitUsageInfo(BaseModel):
    window: str = Field(..., description='The plan\'s usage window: "five_hour", "seven_day", …')
    utilization: float = Field(..., description="0.0-1.0")
    resets_at: datetime | None = None

    @classmethod
    def of(cls, limit: LimitUsage) -> "LimitUsageInfo":
        return cls(window=limit.window, utilization=limit.utilization, resets_at=limit.resets_at)


class SessionStatusResponse(BaseModel):
    context_id: str
    agent: str
    model: str
    context: ContextUsageInfo | None = Field(
        None, description="null until the session has started."
    )
    limits: list[LimitUsageInfo] = Field(default_factory=list)
    background_tasks: int = 0

    @classmethod
    def of(cls, context_id: str, agent: str, status: SessionStatus) -> "SessionStatusResponse":
        return cls(
            context_id=context_id,
            agent=agent,
            model=status.model,
            context=ContextUsageInfo.of(status.context) if status.context else None,
            limits=[LimitUsageInfo.of(limit) for limit in status.limits],
            background_tasks=status.background_tasks,
        )
