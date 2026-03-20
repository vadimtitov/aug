"""Image generation tool via LiteLLM proxy."""

import logging
from typing import Literal

import httpx
from langchain_core.tools import tool
from openai import AsyncOpenAI

from aug.config import get_settings
from aug.core.tools.output import ImageAttachment, ToolOutput

logger = logging.getLogger(__name__)

ImageSize = Literal["1024x1024", "1792x1024", "1024x1792"]


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
        size:   Image dimensions — "1024x1024" (square), "1792x1024" (landscape),
                or "1024x1792" (portrait). Default: square.
        n:      Number of images to generate (1–3). Default: 1.
    """
    n = max(1, min(n, 3))

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    model = settings.IMAGE_GEN_MODEL

    logger.info("generate_image model=%r size=%s n=%d prompt=%r", model, size, n, prompt[:80])

    try:
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            response_format="url",
            n=n,
        )
    except Exception as e:
        logger.exception("generate_image failed")
        return f"Image generation failed: {e}", ToolOutput(text=f"Image generation failed: {e}")

    urls = [item.url for item in response.data if item.url]
    if not urls:
        return "Image generation did not return a URL.", ToolOutput(
            text="Image generation did not return a URL."
        )

    attachments = []
    for url in urls:
        logger.info("generate_image downloading from %s", url)
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                img_response = await http.get(url)
                img_response.raise_for_status()
            attachments.append(ImageAttachment(data=img_response.content, mime_type="image/png"))
        except Exception as e:
            logger.exception("generate_image download failed")
            return f"Image generated but download failed: {e}", ToolOutput(
                text=f"Image generated but download failed: {e}"
            )

    output = ToolOutput(
        text=f"Generated {len(attachments)} image(s) for: {prompt}",
        attachments=attachments,
    )
    return f"Generated {len(attachments)} image(s) for prompt: {prompt!r}", output
