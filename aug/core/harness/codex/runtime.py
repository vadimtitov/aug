"""CodexRuntime: the one ``codex app-server`` process every Codex profile shares.

One process, because the ChatGPT login (``auth.json`` in CODEX_HOME) refreshes by
rotating its refresh token: two processes refreshing at once would sign each other out.
It starts on first use and stays up; if it dies, or stops answering, the next use starts
it again.  It routes what the server says to the thread it is about, answers Codex's
tool calls through that thread, and keeps what is the account's: the plan's usage
limits and the sign-ins in progress (``login.py`` makes them).  A signed-in login is
refreshed once a day, so it never lapses for lack of use.

Codex is set up here, the same for every thread: no approvals and no sandbox (AUG's
container is the boundary, as on Claude Code), nothing loaded from ``~/.codex`` or the
workspace's AGENTS.md (it is in AUG's prompt), none of Codex's own memory, goals, apps,
plugins or image generation (AUG has its own), the workspace's skills as Codex skills,
and the guard hook on Codex's shell, patch and image tools.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import codex_cli_bin

from aug import __version__
from aug.core.harness.codex import guard
from aug.core.harness.types import LimitUsage
from aug.utils.codex_app_server import (
    METHOD_NOT_FOUND,
    AppServerClosed,
    CodexAppServer,
    CodexRpcError,
    JsonObject,
)
from aug.utils.data import CODEX_DIR, SKILLS_DIR, WORKSPACE_DIR

logger = logging.getLogger(__name__)

# A signed-in login is refreshed this often, so it never lapses for lack of use.
_REFRESH_EVERY_SECONDS = 24 * 60 * 60
# What reaches the app-server's environment, and so every command it runs; AUG's own
# secrets (API keys, DATABASE_URL, ...) never do.  hushed injects the user's.
_ENV_KEEP = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "USER",
    "SHELL",
    "SSL_CERT_FILE",
    "HUSHED_DIR",
)
# Codex's settings, as ``-c key=value`` (TOML values).
_CONFIG = {
    "approval_policy": '"never"',
    "sandbox_mode": '"danger-full-access"',
    "project_doc_max_bytes": "0",
    "skills.bundled.enabled": "false",
    "check_for_update_on_startup": "false",
    "analytics.enabled": "false",
    "feedback.enabled": "false",
    "history.persistence": '"none"',
    "cli_auth_credentials_store": '"file"',
    "web_search": '"live"',
    "memories.generate_memories": "false",
    "features.memories": "false",
    "features.goals": "false",
    "features.apps": "false",
    "features.connectors": "false",
    "features.plugins": "false",
    "features.image_generation": "false",
    "features.remote_control": "false",
    "features.browser_use": "false",
    "features.computer_use": "false",
    # Subagents on multi-agent v2, whatever the model's metadata says: v1 lets a thread
    # drive and read any thread whose id it knows, another chat's included.
    "features.multi_agent": "false",
    "features.multi_agent_v2": "true",
    # The environment is already cut down to _ENV_KEEP; CODEX_HOME (where the login is)
    # is Codex's alone.
    "shell_environment_policy.inherit": '"all"',
    "shell_environment_policy.exclude": '["CODEX_HOME"]',
    # A login shell would reset PATH from /etc/profile, losing AUG's venv.
    "allow_login_shell": "false",
}
# Config files in CODEX_HOME that Codex ranks above AUG's own settings.
_OVERRIDING_CONFIG = ("managed_config.toml", "requirements.toml")
# Codex's names for the plan's usage windows, by length in minutes.
_WINDOW_NAMES = {300: "five_hour", 10080: "seven_day"}


class CodexUnusable(Exception):
    """Codex is running, but AUG can't safely use it."""


class ThreadListener(Protocol):
    """What a chat's thread is told by the runtime.  *context_id*: the chat it is
    for; "" for none."""

    context_id: str

    def on_notification(self, method: str, params: JsonObject) -> None:
        """A notification about this thread (or one of its subagents)."""

    async def on_tool_call(self, params: JsonObject) -> JsonObject:
        """Codex wants an AUG tool run; returns the call's answer."""

    def on_closed(self) -> None:
        """The app-server is gone: whatever was in flight will never finish."""


class CodexRuntime:
    """The app-server process and everything about it that is not one chat's.

    *config* and *env* are added to Codex's settings and environment (tests point Codex
    at a fake API with them).
    """

    def __init__(
        self,
        home: Path = CODEX_DIR,
        *,
        config: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.home = home
        self._config = {**_CONFIG, **(config or {})}
        self._extra_env = dict(env or {})
        self._server: CodexAppServer | None = None
        self._lock = asyncio.Lock()
        self._listeners: dict[str, ThreadListener] = {}
        self._loaded: set[str] = set()
        self._logins: dict[str, asyncio.Future[str | None]] = {}  # sign-ins in progress
        self._limits: dict[str, LimitUsage] = {}
        self._users = 0
        self._keepalive: asyncio.Task[None] | None = None
        self._unsubscribing: dict[str, asyncio.Task[None]] = {}  # by thread id

    # -- life --------------------------------------------------------------------------

    @asynccontextmanager
    async def using(self) -> AsyncIterator[None]:
        """Held by each harness for its whole life; the last one out stops the server."""
        self._users += 1
        if self._keepalive is None:
            self._keepalive = asyncio.create_task(self._refresh_forever(), name="codex-login")
        try:
            yield
        finally:
            self._users -= 1
            if self._users == 0:
                await self.close()

    async def close(self) -> None:
        if self._keepalive is not None:
            self._keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive
            self._keepalive = None
        for task in self._unsubscribing.values():
            task.cancel()
        await asyncio.gather(*self._unsubscribing.values(), return_exceptions=True)
        async with self._lock:  # not while one is starting: it would outlive this
            if self._server is not None:
                await self._stop(self._server)

    async def request(self, method: str, params: JsonObject | None = None) -> JsonObject:
        """Send *method* to the app-server, starting it first if it isn't running.  One
        it does not answer in time means it is stuck: it is stopped, and the next
        request starts it again."""
        server = await self._running()
        try:
            return await server.request(method, params)
        except TimeoutError:
            logger.warning("codex app-server did not answer %s: restarting it", method)
            await self._stop(server)
            raise

    # -- threads -----------------------------------------------------------------------

    def listen(self, thread_id: str, listener: ThreadListener) -> bool:
        """Send what is said about *thread_id* to *listener*.  False, and nothing
        changes, if another chat listens to it: one chat never takes over another's.
        (The same chat on another of its harnesses may: a switched agent version.)"""
        current = self._listeners.get(thread_id)
        if current not in (None, listener) and current.context_id != listener.context_id:
            logger.warning("codex thread is another chat's, not taken over: %s", thread_id)
            return False
        self._listeners[thread_id] = listener
        return True

    def forget(self, thread_id: str, listener: ThreadListener) -> None:
        """Stop sending what is said about *thread_id* to *listener*, if it still
        listens to it."""
        if self._listeners.get(thread_id) is listener:
            del self._listeners[thread_id]

    def is_loaded(self, thread_id: str) -> bool:
        """Whether the running app-server has *thread_id* open (else: resume it)."""
        return thread_id in self._loaded and self._server is not None and self._server.alive

    def mark_loaded(self, thread_id: str) -> None:
        self._loaded.add(thread_id)

    def unload(self, thread_id: str, listener: ThreadListener) -> None:
        """*listener* is done with *thread_id*: stop telling it, and let the app-server
        unload the thread (it is resumed when next used).  Nothing changes if another
        listener has it now."""
        if self._listeners.get(thread_id) not in (None, listener):
            return
        self.forget(thread_id, listener)
        self._loaded.discard(thread_id)
        server = self._server
        if server is None or not server.alive:
            return  # nothing to unload: never start a server just for this
        task = asyncio.create_task(_unsubscribe(server, thread_id), name="codex-unsubscribe")
        self._unsubscribing[thread_id] = task
        task.add_done_callback(lambda done: self._unsubscribed(thread_id, done))

    async def unloaded(self, thread_id: str) -> None:
        """Wait until a let-go of *thread_id* has reached the app-server, so a resume
        sent next is never undone by it."""
        if (task := self._unsubscribing.get(thread_id)) is not None:
            await asyncio.wait({task})

    # -- account -----------------------------------------------------------------------

    @property
    def limits(self) -> tuple[LimitUsage, ...]:
        """The plan's usage windows, as last reported; windows already reset left out."""
        now = datetime.now(UTC)
        return tuple(
            sorted(
                (
                    lim
                    for lim in self._limits.values()
                    if lim.resets_at is None or lim.resets_at > now
                ),
                key=lambda lim: lim.window,
            )
        )

    def retry_at(self) -> datetime | None:
        """When a used-up usage window resets, if one is."""
        resets = [lim.resets_at for lim in self.limits if lim.utilization >= 1 and lim.resets_at]
        return min(resets) if resets else None

    def login_finished(self, login_id: str) -> asyncio.Future[str | None]:
        """Resolves when the sign-in *login_id* ends: None once signed in, else why
        not.  Asked for as soon as the sign-in starts; resolving it ends the wait."""
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._logins[login_id] = future
        future.add_done_callback(lambda _: self._logins.pop(login_id, None))
        return future

    async def refresh_limits(self) -> None:
        """Ask for the plan's usage limits now (they also arrive with every turn)."""
        result = await self.request("account/rateLimits/read")
        self._update_limits(result.get("rateLimits"))

    # -- internals ---------------------------------------------------------------------

    def _unsubscribed(self, thread_id: str, task: asyncio.Task[None]) -> None:
        if self._unsubscribing.get(thread_id) is task:
            del self._unsubscribing[thread_id]

    async def _running(self) -> CodexAppServer:
        async with self._lock:
            if self._server is not None and self._server.alive:
                return self._server
            if self._server is not None:
                await self._stop(self._server)
            self._server = await self._start()
            return self._server

    async def _stop(self, server: CodexAppServer) -> None:
        """Stop *server*.  If it was the running one and still up, whatever was in
        flight on it ends now (one that exited by itself has said so already)."""
        ending = server is self._server and server.alive
        if server is self._server:
            self._server = None
        await server.close()
        if ending:
            self._server_gone()

    async def _start(self) -> CodexAppServer:
        await asyncio.to_thread(self._write_config)
        flags = [arg for key, value in self._config.items() for arg in ("-c", f"{key}={value}")]
        server = CodexAppServer(
            [str(codex_cli_bin.bundled_codex_path()), *flags, "app-server", "--listen", "stdio://"],
            env=self._env(),
            cwd=self.home,
            on_notification=self._on_notification,
            on_request=self._on_request,
            on_exit=self._server_gone,
            client_version=__version__,
        )
        await server.start()
        try:
            await self._trust_guard(server)
            if SKILLS_DIR.is_dir():
                await server.request("skills/extraRoots/set", {"extraRoots": [str(SKILLS_DIR)]})
        except BaseException:
            await server.close()
            raise
        return server

    def _write_config(self) -> None:
        """Codex's config.toml: the guard hook, which Codex only runs from here once
        trusted.  Rewritten at every start, so nothing else written there survives; the
        config layers that would rank above it are removed."""
        self.home.mkdir(parents=True, exist_ok=True)
        for name in _OVERRIDING_CONFIG:
            (self.home / name).unlink(missing_ok=True)
        (self.home / "config.toml").write_text(
            "[[hooks.PreToolUse]]\n"
            f"matcher = {json.dumps(guard.MATCHER)}\n"
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            f"command = {json.dumps(guard.hook_command())}\n"
            f"timeout = {guard.HOOK_TIMEOUT_SECONDS}\n"
        )

    async def _trust_guard(self, server: CodexAppServer) -> None:
        """Trust AUG's guard hook, and only it: an untrusted hook never runs.  Checked
        again after, so a server that won't run the guard is never used."""
        for hook in await _guard_hooks(server):
            if hook.get("trustStatus") != "trusted":
                await server.request(
                    "config/value/write",
                    {
                        "keyPath": f"hooks.state.{json.dumps(hook['key'])}.trusted_hash",
                        "value": hook["currentHash"],
                        "mergeStrategy": "upsert",
                    },
                )
        hooks = await _guard_hooks(server)
        if not hooks or any(h.get("trustStatus") != "trusted" for h in hooks):
            raise CodexUnusable("codex does not run AUG's guard hook")

    def _env(self) -> dict[str, str]:
        env = {name: os.environ[name] for name in _ENV_KEEP if name in os.environ}
        env.update(self._extra_env)
        env["CODEX_HOME"] = str(self.home)
        if (tools := codex_cli_bin.bundled_path_dir()) is not None:  # rg
            env["PATH"] = os.pathsep.join(p for p in (str(tools), env.get("PATH", "")) if p)
        return env

    def _on_notification(self, method: str, params: JsonObject) -> None:
        match method:
            case "account/rateLimits/updated":
                self._update_limits(params.get("rateLimits"))
                return
            case "account/login/completed":
                future = self._logins.get(str(params.get("loginId")))
                if future is not None and not future.done():
                    future.set_result(None if params.get("success") else str(params.get("error")))
                return
            case "thread/closed":
                self._loaded.discard(str(params.get("threadId")))
        thread_id = params.get("threadId") or (params.get("thread") or {}).get("id")
        listener = self._listeners.get(thread_id) if isinstance(thread_id, str) else None
        if listener is not None:
            listener.on_notification(method, params)

    async def _on_request(self, method: str, params: JsonObject) -> JsonObject:
        match method:
            case "item/tool/call":
                listener = self._listeners.get(str(params.get("threadId")))
                if listener is None:
                    raise CodexRpcError(-32603, "no AUG chat is listening to this thread")
                return await listener.on_tool_call(params)
            case "item/tool/requestUserInput":
                return {"answers": {}}  # nobody answers mid-turn; Codex goes on without
            case "currentTime/read":
                return {"currentTimeAt": int(time.time())}
        logger.warning("codex asked for %s, which AUG does not answer", method)
        raise CodexRpcError(METHOD_NOT_FOUND, f"AUG does not handle {method}")

    def _server_gone(self) -> None:
        """The server is gone (exited, or stopped): what was in flight on it ends."""
        self._loaded.clear()
        for listener in list(self._listeners.values()):
            listener.on_closed()
        for future in list(self._logins.values()):
            if not future.done():
                future.set_result("the Codex process exited")

    def _update_limits(self, snapshot: Any) -> None:
        if not isinstance(snapshot, dict) or snapshot.get("limitId") not in (None, "codex"):
            return
        for key in ("primary", "secondary"):
            window = snapshot.get(key)
            if not isinstance(window, dict) or window.get("usedPercent") is None:
                continue
            minutes = window.get("windowDurationMins")
            name = _WINDOW_NAMES.get(minutes, f"{minutes}_minutes" if minutes else key)
            resets = window.get("resetsAt")
            self._limits[name] = LimitUsage(
                window=name,
                utilization=float(window["usedPercent"]) / 100,
                resets_at=datetime.fromtimestamp(resets, UTC)
                if isinstance(resets, int | float)
                else None,
            )

    async def _refresh_forever(self) -> None:
        while True:
            try:
                if (self.home / "auth.json").exists():
                    result = await self.request("account/read", {"refreshToken": True})
                    if result.get("account") is None:
                        logger.warning("codex login could not be refreshed: signed out")
                    else:
                        await self.refresh_limits()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("codex login refresh failed: %r", e)
            await asyncio.sleep(_REFRESH_EVERY_SECONDS)


async def _unsubscribe(server: CodexAppServer, thread_id: str) -> None:
    with contextlib.suppress(CodexRpcError, AppServerClosed, TimeoutError):
        await server.request("thread/unsubscribe", {"threadId": thread_id})


async def _guard_hooks(server: CodexAppServer) -> list[JsonObject]:
    """The hooks Codex has loaded that are AUG's guard."""
    listing = await server.request("hooks/list", {"cwds": [str(WORKSPACE_DIR)]})
    return [
        hook
        for entry in listing.get("data") or []
        for hook in entry.get("hooks") or []
        if hook.get("command") == guard.hook_command()
    ]


# The one runtime every Codex profile shares (see the module docstring).
codex_runtime = CodexRuntime()
