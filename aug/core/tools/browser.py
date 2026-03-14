"""Browser tool — remote-controlled Chromium via browser-use and CDP."""

import asyncio
import logging
import os
import socket
from contextvars import ContextVar
from urllib.parse import urlparse, urlunparse

from browser_use import Agent, Browser
from browser_use.agent.views import AgentOutput
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm import ChatOpenAI as BrowserLLM
from langchain_core.tools import tool

from aug.config import get_settings
from aug.utils.user_settings import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"

# Callers (e.g. Telegram handler) can set this to an asyncio.Queue[str] before
# invoking the graph. The browser tool will push a human-readable status string
# into the queue on every browser-use step so the caller can show live progress.
# A None sentinel is pushed when the tool finishes (success or failure).
browser_progress_queue: ContextVar[asyncio.Queue[str | None] | None] = ContextVar(
    "browser_progress_queue", default=None
)


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


@tool
async def browser(task: str, secrets: dict[str, str] | None = None) -> str:
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
        return "Browser tool is not available — BROWSER_CDP_URL is not configured."

    sensitive_data = _resolve_secrets(secrets)

    queue = browser_progress_queue.get()

    async def _step_callback(state: BrowserStateSummary, output: AgentOutput, step: int) -> None:
        if queue is None:
            return
        goal = output.next_goal or ""
        netloc = urlparse(state.url).netloc or state.url
        text = f"Step {step} · {netloc}"
        if goal:
            text += f"\n{goal}"
        await queue.put(text)

    b = Browser(cdp_url=_resolve_cdp_url(cdp_url))
    try:
        agent = Agent(
            task=task,
            llm=_llm(),
            browser=b,
            sensitive_data=sensitive_data or None,
            register_new_step_callback=_step_callback,
            extend_system_message=(
                "Only perform actions explicitly required by the task. "
                "Do not modify, remove, or interact with anything not mentioned in the task."
            ),
        )
        history = await agent.run()
        if history.is_done() and history.final_result():
            return history.final_result()
        errors = history.errors() if history.has_errors() else []
        error_summary = "; ".join(str(e) for e in errors[-3:]) if errors else "unknown reason"
        return (
            f"Browser task did NOT complete successfully after {history.number_of_steps()} steps. "
            f"Errors: {error_summary}. Do NOT assume success — tell the user it failed and why."
        )
    except Exception as e:
        logger.error("browser tool failed: %s", e)
        return f"Browser task failed: {e}"
    finally:
        await b.stop()
        if queue is not None:
            await queue.put(None)


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


def _is_ip_or_localhost(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False
