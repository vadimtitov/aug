"""Browser tool — remote-controlled Chromium via browser-use and CDP."""

import base64
import logging
import mimetypes
import os
import socket
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from browser_use import Agent, Browser
from browser_use.agent.views import AgentOutput
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm import ChatOpenAI as BrowserLLM
from langchain_core.tools import tool

from aug.config import get_settings
from aug.core.events import send_tool_progress_update
from aug.core.prompts import BROWSER_TASK_CONSTRAINTS
from aug.core.tools.output import Attachment, FileAttachment, ImageAttachment, ToolOutput
from aug.utils.user_settings import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.1"
_DOWNLOADS_DIR = "/app/browser-downloads"


def _model() -> str:
    return get_setting("tools", "browser", "model", default=_DEFAULT_MODEL)


def _llm() -> BrowserLLM:
    settings = get_settings()
    return BrowserLLM(
        model=_model(),
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        frequency_penalty=None,
    )


@tool(response_format="content_and_artifact")
async def browser(
    task: str, secrets: dict[str, str] | None = None
) -> tuple[str, ToolOutput | None]:
    """Control a web browser to complete a task.

    Can navigate websites, click buttons, fill forms, log in, and extract
    information from any page. Use for tasks that require interacting with a
    live website — ordering food, checking prices, filling in forms, logging
    into services, scraping JS-rendered content, etc.

    For tasks that need credentials, use {placeholder} syntax in the task and
    pass the corresponding env var names in secrets. The real values are injected
    directly into the browser without ever appearing in the LLM context.

    Example:
        task="Log into github.com with username {username} and password {password}"
        secrets={"username": "GITHUB_USER", "password": "GITHUB_PASSWORD"}

    Args:
        task: Plain-language description. Use {placeholder} for any sensitive values.
        secrets: Map of {placeholder: env_var_name} for credentials or other secrets.
    """
    cdp_url = get_settings().BROWSER_CDP_URL
    if not cdp_url:
        return "Browser tool is not available — BROWSER_CDP_URL is not configured.", None

    sensitive_data = _resolve_secrets(secrets)

    async def _step_callback(state: BrowserStateSummary, output: AgentOutput, step: int) -> None:
        goal = output.next_goal or ""
        netloc = urlparse(state.url).netloc or state.url
        text = f"Step {step} · {netloc}"
        if goal:
            text += f"\n{goal}"
        await send_tool_progress_update(text)

    downloads_dir = Path(_DOWNLOADS_DIR)
    before = set(downloads_dir.iterdir()) if downloads_dir.exists() else set()
    b = Browser(cdp_url=_resolve_cdp_url(cdp_url), downloads_path=_DOWNLOADS_DIR)
    try:
        agent = Agent(
            task=task,
            llm=_llm(),
            browser=b,
            sensitive_data=sensitive_data or None,
            register_new_step_callback=_step_callback,
            extend_system_message=BROWSER_TASK_CONSTRAINTS,
            use_vision="auto",
        )
        history = await agent.run()
        if history.is_done() and history.final_result():
            after = set(downloads_dir.iterdir()) if downloads_dir.exists() else set()
            new_files = [p for p in (after - before) if p.is_file()]
            logger.info("browser new files: %s", new_files)
            attachments: list[Attachment] = [_path_to_attachment(str(p)) for p in new_files]
            screenshots = history.screenshots(n_last=1, return_none_if_not_screenshot=False)
            logger.info("browser screenshots: %d", len(screenshots))
            if screenshots:
                attachments.append(
                    ImageAttachment(data=base64.b64decode(screenshots[-1]), mime_type="image/png")
                )
            output = ToolOutput(text=history.final_result(), attachments=attachments)
            return str(output), output
        errors = history.errors() if history.has_errors() else []
        error_summary = "; ".join(str(e) for e in errors[-3:]) if errors else "unknown reason"
        msg = (
            f"Browser task did NOT complete successfully after {history.number_of_steps()} steps. "
            f"Errors: {error_summary}. Do NOT assume success — tell the user it failed and why."
        )
        return msg, None
    except Exception as e:
        logger.error("browser tool failed: %s", e)
        return f"Browser task failed: {e}", None
    finally:
        await b.stop()


def _resolve_secrets(secrets: dict[str, str] | None) -> dict[str, str]:
    """Resolve env var names to their actual values for browser-use sensitive_data.

    Missing env vars are skipped with a warning so the agent still runs.
    """
    if not secrets:
        return {}
    resolved = {}
    for placeholder, env_var in secrets.items():
        value = os.environ.get(env_var)
        if value:
            resolved[placeholder] = value
        else:
            logger.warning("browser secrets: env var %r not found", env_var)
    return resolved


def _resolve_cdp_url(cdp_url: str) -> str:
    """Replace the hostname in a CDP URL with its resolved IP address.

    Chrome's CDP HTTP endpoint rejects requests whose Host header is not an IP
    or 'localhost'. Inside Docker, service names resolve to container IPs, so we
    pre-resolve the hostname here to satisfy that check.
    """
    parsed = urlparse(cdp_url)
    if parsed.hostname and not _is_ip_or_localhost(parsed.hostname):
        ip = socket.gethostbyname(parsed.hostname)
        resolved = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        return urlunparse(resolved)
    return cdp_url


def _path_to_attachment(path: str) -> Attachment:
    data = Path(path).read_bytes()
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    if mime.startswith("image/"):
        return ImageAttachment(data=data, mime_type=mime)
    return FileAttachment(data=data, filename=Path(path).name)


def _is_ip_or_localhost(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False
