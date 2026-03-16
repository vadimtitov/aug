"""Rich tool output types.

Tools that want to return non-text content alongside their text result
can return a ToolOutput instead of a plain string. The interface layer
handles attachments; the LLM only sees the text.

Usage:
    @tool
    async def my_tool(query: str) -> ToolOutput:
        data = fetch_image(...)
        return ToolOutput(
            text="Here is the image.",
            attachments=[ImageAttachment(data=data)],
        )
"""

from dataclasses import dataclass, field


@dataclass
class ImageAttachment:
    data: bytes
    mime_type: str = "image/jpeg"
    caption: str = ""


@dataclass
class FileAttachment:
    data: bytes
    filename: str
    caption: str = ""


Attachment = ImageAttachment | FileAttachment


@dataclass
class ToolOutput:
    text: str
    attachments: list[Attachment] = field(default_factory=list)

    def __str__(self) -> str:
        """Return only the text — this is what the LLM sees."""
        return self.text
