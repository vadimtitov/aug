"""Media helpers: image normalisation and media-type guessing.

Images are normalised once, when they arrive, so every model call after that sends the
same small bytes.  Claude rejects any image over 3000 px once a request holds more than
20 images (and the whole history counts), and 1568 px on the long edge keeps an image
at about 1.8k tokens.
"""

import io
import mimetypes
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

MAX_IMAGE_EDGE = 1568
JPEG_QUALITY = 85

# Formats sent as they are when they are small enough.  Anything else Pillow can open
# (HEIC, TIFF, BMP…) is re-encoded to JPEG, which every model accepts.
_PASSTHROUGH_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def normalise_image(data: bytes, media_type: str) -> tuple[bytes, str]:
    """Return *data* as an image a model accepts: JPEG at most MAX_IMAGE_EDGE px on its
    long edge, or the original bytes when they already are within bounds.

    Data Pillow cannot read is returned unchanged.  Animated images are left alone too:
    re-encoding would keep only their first frame.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return data, media_type
    if getattr(image, "is_animated", False):
        return data, media_type
    if media_type in _PASSTHROUGH_TYPES and max(image.size) <= MAX_IMAGE_EDGE:
        return data, media_type

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
