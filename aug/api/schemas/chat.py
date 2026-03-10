"""Chat endpoint schemas."""

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    file_id: str
    mime_type: str | None = None


class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="UUID of the conversation thread.")
    message: str = Field(..., description="User message text.")
    agent: str = Field(default="default", description="Agent version string.")
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
