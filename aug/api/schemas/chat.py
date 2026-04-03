"""Chat endpoint schemas."""

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    file_id: str
    mime_type: str | None = None


class ChatRequest(BaseModel):
    thread_id: str = Field(
        ..., max_length=128, pattern=r"^[\w\-]+$", description="Conversation thread ID."
    )
    message: str = Field(..., max_length=32_000, description="User message text.")
    agent: str = Field(default="default", max_length=64, description="Agent version string.")
    attachments: list[Attachment] = Field(default_factory=list)


class ToolCallInfo(BaseModel):
    name: str
    input: dict = Field(default_factory=dict)
    output: dict | None = None


class ChatResponse(BaseModel):
    thread_id: str
    agent: str
    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    decision: str = Field(
        ...,
        pattern=r"^(approved_once|approved_always|denied)$",
        description="Approval decision: approved_once | approved_always | denied",
    )
    agent: str = Field(
        default="default",
        max_length=64,
        description="Agent version — must match the paused thread's agent.",
    )
    sender_id: str = Field(
        default="",
        max_length=128,
        description="Sender ID to attribute the resume to (e.g. thread_id for REST).",
    )
