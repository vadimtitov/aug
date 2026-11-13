"""Image generation and editing tools via LiteLLM proxy."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.tools import tool
from openai import AsyncOpenAI

from aug.config import get_settings
from aug.core.tools.context import current_tool_context
from aug.utils.file_settings import load_settings
from aug.utils.storage import file_store

logger = logging.getLogger(__name__)

ImageSize = Literal["1024x1024", "1536x1024", "1024x1536", "auto"]


@tool
async def generate_image(
    prompt: str,
    size: ImageSize = "1024x1024",
    n: int = 1,
) -> str:
    """Generate one or more images from a text description and send them to the user.

    Use this when the user asks to create, draw, generate, or visualise something.

    Args:
        prompt: Detailed description of the image to generate.
        size:   Image dimensions — "1024x1024" (square), "1536x1024" (landscape),
                "1024x1536" (portrait), or "auto". Default: square.
        n:      Number of images to generate (1–3). Default: 1.
    """
    n = max(1, min(n, 3))

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    model = load_settings().tools.image_gen.model

    logger.info("generate_image model=%r size=%s n=%d prompt=%r", model, size, n, prompt[:80])

    try:
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
        )
    except Exception as e:
        logger.exception("generate_image failed")
        return f"Image generation failed: {e}"

    images: list[bytes] = []
    for item in response.data or []:
        if item.b64_json:
            images.append(base64.b64decode(item.b64_json))
        elif item.url:
            try:
                images.append(await _download(item.url))
            except Exception as e:
                logger.exception("generate_image download failed")
                return f"Image generated but download failed: {e}"

    if not images:
        return "Image generation did not return any data."

    paths = [await _send_image(data, f"generated-{i + 1}.png") for i, data in enumerate(images)]
    return (
        f"Generated {len(images)} image(s) for prompt {prompt!r} and sent them to the user. "
        f"Saved at: {', '.join(paths)}"
    )


@tool
async def edit_image(
    source_path: str,
    prompt: str,
    size: ImageSize = "1024x1024",
) -> str:
    """Edit or transform an existing image based on a text instruction, and send the
    result to the user.

    Use this when the user wants to modify, restyle, or transform an image they
    have already sent (its path is in the "[Image from user: …]" note) or one you
    generated earlier.

    Args:
        source_path: Absolute path to the source image on disk.
        prompt:      Description of the desired edit or transformation.
        size:        Output dimensions — "1024x1024", "1536x1024", "1024x1536", or "auto".
                     Default: "1024x1024".
    """
    resolved = Path(source_path).resolve()
    if not resolved.exists():
        return f"Source image not found at path: {source_path}"

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    model = load_settings().tools.image_gen.model

    logger.info(
        "edit_image model=%r size=%s path=%r prompt=%r", model, size, source_path, prompt[:80]
    )

    try:
        image_bytes = await asyncio.to_thread(resolved.read_bytes)
        filename = resolved.name
        response = await client.images.edit(
            model=model,
            image=(filename, image_bytes),
            prompt=prompt,
            size=size,
            n=1,
        )
    except Exception as e:
        logger.exception("edit_image failed")
        return f"Image editing failed: {e}"

    item = response.data[0] if response.data else None
    if item is None:
        return "Image editing did not return a result."

    if item.b64_json:
        image_data = base64.b64decode(item.b64_json)
    elif item.url:
        try:
            image_data = await _download(item.url)
        except Exception as e:
            logger.exception("edit_image download failed")
            return f"Image edited but download failed: {e}"
    else:
        return "Image editing returned no URL or data."

    path = await _send_image(image_data, f"{resolved.stem}-edited.png")
    return f"Edited the image using prompt {prompt!r} and sent it to the user. Saved at: {path}"


async def _send_image(data: bytes, filename: str) -> str:
    """Store *data* in the outbox and send it to the user; its stored path."""
    stored = await file_store.save_outbound(data, filename, "image/png")
    ctx = current_tool_context()
    if ctx is not None:
        await ctx.send_file(stored)
    return str(stored.path)


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.get(url)
        response.raise_for_status()
    return response.content
