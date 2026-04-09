"""Image description tool factory.

Creates a tool that uses a dedicated vision model to describe images for agents
that cannot process image inputs directly.
"""

import asyncio
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from aug.core.llm import build_chat_model
from aug.core.prompts import IMAGE_DESCRIPTION_PROMPT


def make_describe_image_tool(model: str) -> BaseTool:
    """Return a describe_image tool that uses *model* for vision inference."""
    _llm = build_chat_model(model)

    @tool
    async def describe_image(path: str, question: str) -> str:
        """Understand the content of an image saved on disk.

        Use this tool to understand what an image contains so you can respond
        naturally to the user — do NOT relay the description back verbatim.
        Pass the image path exactly as it appears in the [[img:path|mime]] marker
        and a specific question about what you need to know.

        Args:
            path: Absolute path to the image file on disk.
            question: Specific question about the image to answer.
        """
        try:
            data = await _read_image(path)
        except FileNotFoundError:
            return f"Image not found: {path}"

        mime_type = _mime_from_path(path)
        encoded = base64.b64encode(data).decode()
        image_url = f"data:{mime_type};base64,{encoded}"

        messages = [
            SystemMessage(content=IMAGE_DESCRIPTION_PROMPT),
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ]
            ),
        ]
        try:
            response = await _llm.ainvoke(messages, config={"callbacks": []})
            return str(response.content)
        except Exception as exc:
            return f"Error describing image: {exc}"

    return describe_image


async def _read_image(path: str) -> bytes:
    return await asyncio.to_thread(Path(path).read_bytes)


def _mime_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
