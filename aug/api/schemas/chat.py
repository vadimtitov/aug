"""Chat endpoint schemas."""

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    file_id: str
    mime_type: str | None = None


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


class ChatResponse(BaseModel):
    context_id: str
    agent: str
    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


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
