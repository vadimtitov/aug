"""Media helpers: image normalisation and media-type guessing.

Images are normalised once, when they arrive, so every model call after that sends the
same small bytes.  Claude rejects any image over 3000 px once a request holds more than
20 images (and the whole history counts), and 1568 px on the long edge keeps an image
at about 1.8k tokens.
"""

import io
import mimetypes
import re
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

MAX_IMAGE_EDGE = 1568
# Under Anthropic's 5 MB per-image limit with room to spare; bigger is re-encoded.
MAX_IMAGE_BYTES = 3_500_000
JPEG_QUALITY = 85

# Image types a model accepts inline.  Sent as they are when small enough; anything
# else Pillow can open (HEIC, TIFF, BMP…) is re-encoded to JPEG.
MODEL_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# How an image sits in LangGraph history: expanded into the image on each model call.
IMAGE_MARKER = re.compile(r"\[\[img:([^|\]]+)\|([^\]]+)\]\]")
# What one normalised image costs in a request, roughly (1568 px long edge).
IMAGE_TOKENS = 1600


def normalise_image(data: bytes | bytearray, media_type: str) -> tuple[bytes | bytearray, str]:
    """Return *data* as an image a model accepts: JPEG at most MAX_IMAGE_EDGE px on its
    long edge, or the original bytes when they already are within bounds.

    The bytes decide, not *media_type*: a HEIC sent as "image/jpeg" is still converted,
    and what passes through is labelled with its real type.  Data Pillow cannot read
    (or will not, being too large to decode safely) is returned unchanged; so are
    animated images, as re-encoding would keep only their first frame.
    """
    try:
        image = Image.open(io.BytesIO(data))
        actual = Image.MIME.get(image.format or "", media_type)
        if getattr(image, "is_animated", False):
            return data, actual
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return data, media_type
    if (
        actual in MODEL_IMAGE_TYPES
        and max(image.size) <= MAX_IMAGE_EDGE
        and len(data) <= MAX_IMAGE_BYTES
    ):
        return data, actual

    image = ImageOps.exif_transpose(image)
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        flattened = Image.new("RGB", image.size, (255, 255, 255))
        flattened.paste(image, mask=image.getchannel("A"))
        image = flattened
    elif image.mode != "RGB":
        image = image.convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue(), "image/jpeg"


def image_marker(path: Path, media_type: str) -> str:
    return f"[[img:{path}|{media_type}]]"


def guess_media_type(filename: str, default: str = "application/octet-stream") -> str:
    """The media type for *filename*, from its extension."""
    if Path(filename).suffix.lower() in (".heic", ".heif"):
        return "image/heic"  # not in every system's mime table
    return mimetypes.guess_type(filename)[0] or default


def extension_for(media_type: str) -> str:
    """A file extension (with the dot) for *media_type*, or "" if none is known."""
    if media_type == "image/jpeg":
        return ".jpg"  # mimetypes may answer ".jpe"
    return mimetypes.guess_extension(media_type) or ""
