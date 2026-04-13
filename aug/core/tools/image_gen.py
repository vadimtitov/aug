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
from aug.core.tools.output import ImageAttachment, ToolOutput
from aug.utils.user_settings import get_setting

_DEFAULT_IMAGE_GEN_MODEL = "gpt-image-1.5"

logger = logging.getLogger(__name__)

ImageSize = Literal["1024x1024", "1536x1024", "1024x1536", "auto"]


def _get_image_gen_model() -> str:
    return get_setting("tools", "image_gen", "model") or _DEFAULT_IMAGE_GEN_MODEL


@tool(response_format="content_and_artifact")
async def generate_image(
    prompt: str,
    size: ImageSize = "1024x1024",
    n: int = 1,
) -> tuple[str, ToolOutput]:
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
    model = _get_image_gen_model()

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
        return f"Image generation failed: {e}", ToolOutput(text=f"Image generation failed: {e}")

    attachments = []
    for item in response.data:
        if item.b64_json:
            attachments.append(
                ImageAttachment(data=base64.b64decode(item.b64_json), mime_type="image/png")
            )
        elif item.url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    img_response = await http.get(item.url)
                    img_response.raise_for_status()
                attachments.append(
                    ImageAttachment(data=img_response.content, mime_type="image/png")
                )
            except Exception as e:
                logger.exception("generate_image download failed")
                return f"Image generated but download failed: {e}", ToolOutput(
                    text=f"Image generated but download failed: {e}"
                )

    if not attachments:
        return "Image generation did not return any data.", ToolOutput(
            text="Image generation did not return any data."
        )

    output = ToolOutput(
        text=f"Generated {len(attachments)} image(s) for: {prompt}",
        attachments=attachments,
    )
    return f"Generated {len(attachments)} image(s) for prompt: {prompt!r}", output


@tool(response_format="content_and_artifact")
async def edit_image(
    source_path: str,
    prompt: str,
    size: ImageSize = "1024x1024",
) -> tuple[str, ToolOutput]:
    """Edit or transform an existing image based on a text instruction.

    Use this when the user wants to modify, restyle, or transform an image they
    have already sent. The source image must have been saved to disk (i.e. you
    know its path from a prior [[img:...]] reference).

    Args:
        source_path: Absolute path to the source image on disk.
        prompt:      Description of the desired edit or transformation.
        size:        Output dimensions — "1024x1024", "1536x1024", "1024x1536", or "auto".
                     Default: "1024x1024".
    """
    resolved = Path(source_path).resolve()
    if not resolved.exists():
        msg = f"Source image not found at path: {source_path}"
        return msg, ToolOutput(text=msg)

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    model = _get_image_gen_model()

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
        return f"Image editing failed: {e}", ToolOutput(text=f"Image editing failed: {e}")

    item = response.data[0] if response.data else None
    if item is None:
        msg = "Image editing did not return a result."
        return msg, ToolOutput(text=msg)

    if item.b64_json:
        image_data = base64.b64decode(item.b64_json)
    elif item.url:
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                img_response = await http.get(item.url)
                img_response.raise_for_status()
            image_data = img_response.content
        except Exception as e:
            logger.exception("edit_image download failed")
            return f"Image edited but download failed: {e}", ToolOutput(
                text=f"Image edited but download failed: {e}"
            )
    else:
        msg = "Image editing returned no URL or data."
        return msg, ToolOutput(text=msg)

    output = ToolOutput(
        text=f"Edited image for: {prompt}",
        attachments=[ImageAttachment(data=image_data, mime_type="image/png")],
    )
    return f"Edited image using prompt: {prompt!r}", output
