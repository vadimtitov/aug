"""Tool for sending a response as a file to the user."""

import logging
from pathlib import Path

from langchain_core.tools import tool

from aug.core.tools.output import FileAttachment, ToolOutput

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("/app/data/respond_with_file")


@tool(response_format="content_and_artifact")
def respond_with_file(
    filename: str,
    content: str = "",
    filepath: str = "",
) -> tuple[str, ToolOutput]:
    """Send a file to the user. Use when a response is better consumed as a file —
    long text, structured data, a generated document, a PDF, etc.

    Provide either:
    - content: raw text to write into a new file (requires filename with extension).
    - filepath: absolute path to an existing file inside the container (e.g. a file
      written by run_bash). The filename parameter is used as the download name.

    Args:
        filename: Download filename including extension (e.g. "report.pdf", "jobs.csv").
        content: Text content to send as a file. Use when generating content directly.
        filepath: Absolute path to an existing file to send (e.g. "/tmp/report.pdf").
    """
    if not content and not filepath:
        return "Error: provide either content or filepath.", ToolOutput(
            text="Error: provide either content or filepath."
        )

    if filepath:
        path = Path(filepath)
        if not path.exists():
            return f"Error: file not found at {filepath}.", ToolOutput(
                text=f"Error: file not found at {filepath}."
            )
        data = path.read_bytes()
        logger.info("respond_with_file sending %s as %s (%d bytes)", filepath, filename, len(data))
    else:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / filename
        path.write_text(content, encoding="utf-8")
        data = content.encode("utf-8")
        logger.info("respond_with_file saved %s (%d bytes)", path, len(data))

    output = ToolOutput(
        text=f"Sent as file: {filename}",
        attachments=[FileAttachment(data=data, filename=filename)],
    )
    return f"File '{filename}' sent to the user.", output
