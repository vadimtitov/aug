"""Chat endpoint schemas."""

from pydantic import BaseModel, Field


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
