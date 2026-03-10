"""Thread management schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateThreadRequest(BaseModel):
    agent: str = Field(default="default", description="Agent version to bind this thread to.")


class ThreadMetadata(BaseModel):
    thread_id: str
    agent_version: str
    created_at: datetime
    updated_at: datetime


class MessageRecord(BaseModel):
    role: str  # "human" | "ai" | "tool"
    content: str
    created_at: datetime


class ThreadDetail(BaseModel):
    thread_id: str
    agent_version: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRecord] = Field(default_factory=list)
