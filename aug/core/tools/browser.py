"""Browser tool — remote-controlled Chromium via browser-use and CDP."""

import asyncio
import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from browser_use import ActionResult, Agent, Browser, Tools
from browser_use.agent.views import AgentOutput
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm import ChatOpenAI as BrowserLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolArg

from aug.config import get_settings
from aug.core.events import send_tool_progress_update
from aug.core.llm import build_chat_model
from aug.core.prompts import (
    ASK_HUMAN_CAPTCHA_DESCRIPTION,
    BROWSER_TASK_CONSTRAINTS,
    CAPTCHA_HUMAN_RESPONSE_TEMPLATE,
    CAPTCHA_HUMAN_UNREADABLE,
    CAPTCHA_TRANSCRIPTION_PROMPT,
)
from aug.core.run import AGENT_RUN_CONFIG_KEY, MessageContent
from aug.core.tools.output import Attachment, FileAttachment, ImageAttachment, ToolOutput
from aug.utils.cdp import resolve_cdp_url
from aug.utils.file_settings import load_settings

logger = logging.getLogger(__name__)

_DOWNLOADS_DIR = "/app/browser-downloads"

# Vision model that actually reads the captcha. Hardcoded (not a setting) per the
# project's config philosophy — this is the established vision model and there is
# no immediate reason for it to vary independently. The browser agent is told a
# human does this; nothing in its context reveals a model is involved.
_CAPTCHA_VISION_MODEL = "gemini-2.5-pro"

# Runs in the page to extract the captcha at native resolution. It reads the
# ALREADY-LOADED <img> (painting it to a canvas) rather than re-fetching the URL —
# captcha endpoints typically regenerate a new challenge on each GET, so a re-fetch
# would return a different image than the one the session expects. Pierces open
# shadow roots and matches the image by src/alt/id/class. Returns a PNG data URL, or
# null if no captcha <img> is found (caller falls back to a full-page screenshot).
_CAPTCHA_EXTRACT_JS = """
(() => {
  const imgs = [];
  const walk = (root) => {
    root.querySelectorAll('*').forEach((el) => {
      if (el.tagName === 'IMG') imgs.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
    });
  };
  walk(document);
  const looksLikeCaptcha = (i) =>
    /captcha|\\u043a\\u0430\\u043f\\u0447|\\u0441\\u0438\\u043c\\u0432\\u043e\\u043b/i.test(
      [i.src, i.alt, i.id, i.className].join(' ')
    );
  const img = imgs.find(looksLikeCaptcha);
  if (!img || !img.complete || !img.naturalWidth) return null;
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);
  try {
    return canvas.toDataURL('image/png');
  } catch (e) {
    return null;
  }
})()
"""


@tool(response_format="content_and_artifact")
async def browser(
    task: str,
    secrets: dict[str, str] | None = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
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

    Give the browser the COMPLETE goal in one task and let it work autonomously —
    do not decompose a job into low-level browser calls. In particular, the browser
    solves CAPTCHAs on its own: never use this tool just to "find", "extract",
    "download", or "screenshot" a CAPTCHA image, and never try to read or solve a
    CAPTCHA yourself. If a page has a CAPTCHA, simply include the surrounding goal
    (e.g. "log in with these credentials") in the task; the browser will read and
    fill the CAPTCHA internally as part of completing it.

    Args:
        task: Plain-language description. Use {placeholder} for any sensitive values.
        secrets: Map of {placeholder: env_var_name} for credentials or other secrets.
    """
    cdp_url = get_settings().BROWSER_CDP_URL
    if not cdp_url:
        return "Browser tool is not available — BROWSER_CDP_URL is not configured.", None

    sensitive_data = _resolve_secrets(secrets)
    run = (config or {}).get("configurable", {}).get(AGENT_RUN_CONFIG_KEY)

    # Forward reference — populated after Agent is created below.
    agent_ref: list[Agent] = []

    async def _step_callback(state: BrowserStateSummary, output: AgentOutput, step: int) -> None:
        # Drain any user messages that arrived mid-run and inject them into the
        # browser agent's context so it can adapt without losing progress.
        if run and agent_ref and run.pending_agent_injection.qsize() > 0:
            messages: list[MessageContent] = []
            while True:
                try:
                    messages.append(run.pending_agent_injection.get_nowait())
                except asyncio.QueueEmpty:
                    break
            agent_ref[0].add_new_task(_format_injected_messages(messages))

        # Lead with the goal (what the agent is about to do) — far more useful than
        # the netloc, which is already shown in the Browser(task) header.
        goal = (output.next_goal or "").strip().replace("\n", " ")
        netloc = urlparse(state.url).netloc or state.url
        text = f"Step {step} · {goal}" if goal else f"Step {step} · {netloc}"
        await send_tool_progress_update(text)

    async def _should_stop() -> bool:
        return run is not None and run.user_requested_stop.is_set()

    downloads_dir = Path(_DOWNLOADS_DIR)
    before = set(downloads_dir.iterdir()) if downloads_dir.exists() else set()
    b = Browser(cdp_url=resolve_cdp_url(cdp_url), downloads_path=_DOWNLOADS_DIR)
    try:
        agent = Agent(
            task=task,
            llm=_llm(),
            browser=b,
            tools=_build_tools(),
            sensitive_data=sensitive_data or None,
            register_new_step_callback=_step_callback,
            register_should_stop_callback=_should_stop,
            extend_system_message=BROWSER_TASK_CONSTRAINTS,
            use_vision="auto",
        )
        agent_ref.append(agent)
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

        if run and run.user_requested_stop.is_set():
            return _stopped_summary(agent), None

        errors = history.errors() if history.has_errors() else []
        error_summary = "; ".join(str(e) for e in errors[-3:]) if errors else "unknown reason"
        msg = (
            f"Browser task did NOT complete successfully after {history.number_of_steps()} steps. "
            f"Errors: {error_summary}. Do NOT assume success — tell the user it failed and why."
        )
        return msg, None
    except Exception as e:
        logger.exception("browser tool failed")
        return f"Browser task failed: {e}", None
    finally:
        await b.stop()


def _llm() -> BrowserLLM:
    settings = get_settings()
    return BrowserLLM(
        model=load_settings().tools.browser.model,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        frequency_penalty=None,
    )


def _build_tools() -> Tools:
    """Browser-use's default action set plus a captcha-reading action.

    The action is framed to the browser agent as asking a human assistant, so it
    confidently hands off image-text captchas instead of stalling. A vision model
    does the transcription; the agent never sees that.
    """
    tools = Tools()
    vision_llm = build_chat_model(_CAPTCHA_VISION_MODEL)

    @tools.action(ASK_HUMAN_CAPTCHA_DESCRIPTION)
    async def ask_human_to_solve_captcha(browser_session: BrowserSession) -> ActionResult:
        return await _solve_captcha(vision_llm, browser_session)

    return tools


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


def _path_to_attachment(path: str) -> Attachment:
    data = Path(path).read_bytes()
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    if mime.startswith("image/"):
        return ImageAttachment(data=data, mime_type=mime)
    return FileAttachment(data=data, filename=Path(path).name)


def _format_injected_messages(messages: list[MessageContent]) -> str:
    """Convert queued user messages to a string for add_new_task().

    Extracts text from each message (multimodal content loses non-text parts,
    which is acceptable — images can't be injected as browser task context).
    """
    parts = []
    for msg in messages:
        if isinstance(msg, str):
            parts.append(msg)
        else:
            text = "\n".join(b["text"] for b in msg if b.get("type") == "text")
            if text:
                parts.append(text)
    combined = "\n\n".join(parts)
    return f"[User sent this message while you were working]: {combined}"


def _stopped_summary(agent: Agent) -> str:
    """Build a rich handoff string when the browser task is explicitly stopped."""
    history = agent.history
    n = history.number_of_steps()
    parts: list[str] = [f"Browser task was stopped after {n} steps."]

    thoughts = history.model_thoughts()
    if thoughts:
        last = thoughts[-1]
        if last.memory:
            parts.append(
                f"\nProgress summary (browser agent's own accumulated notes):\n{last.memory}"
            )
        if last.next_goal:
            parts.append(f"\nWas about to: {last.next_goal}")

    urls = [u for u in history.urls() if u]
    if urls:
        parts.append(f"\nLast page: {urls[-1]}")

    extracted = history.extracted_content()
    if extracted:
        parts.append("\nContent extracted before stopping:\n" + "\n".join(extracted))

    return "\n".join(parts)


async def _solve_captcha(vision_llm, browser_session: BrowserSession) -> ActionResult:
    """Read the captcha with the vision model and reply in the voice of a human.

    Prefers the raw captcha image extracted from the page at native resolution (far
    clearer than a downscaled page render); falls back to a full-viewport screenshot
    when no captcha <img> can be isolated. Any failure returns an explicit "could not
    read" message so the agent never types a guess.
    """
    try:
        png = await _grab_captcha_image(browser_session)
        source = "image"
        if png is None:
            png = await browser_session.take_screenshot()
            source = "screenshot"
        solution = await _transcribe_captcha(vision_llm, png)
    except Exception as e:
        logger.exception("captcha action failed")
        return ActionResult(
            extracted_content=(
                f"The human could not check the CAPTCHA (error: {e}). Do NOT guess a value."
            ),
            include_in_memory=True,
        )
    logger.info("captcha: source=%s, %d bytes, vision read=%r", source, len(png), solution)
    if not solution or solution.upper() == "UNREADABLE":
        return ActionResult(extracted_content=CAPTCHA_HUMAN_UNREADABLE, include_in_memory=True)
    return ActionResult(
        extracted_content=CAPTCHA_HUMAN_RESPONSE_TEMPLATE.format(solution=solution),
        include_in_memory=True,
    )


async def _grab_captcha_image(browser_session: BrowserSession) -> bytes | None:
    """Extract the loaded captcha <img> from the page as PNG bytes, or None.

    Reads the already-displayed image via canvas (no new network request), so it can't
    trigger captcha regeneration and needs no separate auth/cookies.
    """
    cdp = await browser_session.get_or_create_cdp_session()
    result = await cdp.cdp_client.send.Runtime.evaluate(
        params={"expression": _CAPTCHA_EXTRACT_JS, "returnByValue": True, "awaitPromise": True},
        session_id=cdp.session_id,
    )
    value = (result.get("result") or {}).get("value")
    if not isinstance(value, str) or not value.startswith("data:image"):
        return None
    return base64.b64decode(value.split(",", 1)[1])


async def _transcribe_captcha(vision_llm, png: bytes) -> str:
    image_url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
    messages = [
        SystemMessage(content=CAPTCHA_TRANSCRIPTION_PROMPT),
        HumanMessage(content=[{"type": "image_url", "image_url": {"url": image_url}}]),
    ]
    response = await vision_llm.ainvoke(messages, config={"callbacks": []})
    return str(response.content).strip()
