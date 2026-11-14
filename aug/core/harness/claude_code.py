"""ClaudeCodeHarness: a profile run on the Claude Agent SDK (the Claude Code CLI).

Each chat gets its own CLI process, opened on first use and kept warm; idle ones are
closed after IDLE_SECONDS (the prompt cache's lifetime), never while a turn or
background work runs.  A closed session resumes from Claude's own session file on the
next message.  What a session started with (its prompt, skills, tools and settings) is
fingerprinted; when it changes, the process is restarted and resumes.

The system prompt is Claude Code's own plus AUG's session prompt, built once per
session and kept in its record, like on LangGraph.  Nothing loads from ``~/.claude``:
filesystem settings are off and the workspace's skills load as a local plugin.
AUG's tools reach Claude through one in-process MCP server; Claude's built-in tools are
an allowlist and run without approval, behind the guard hook.
"""

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from langchain_core.tools import BaseTool

from aug.core.harness.base import Harness
from aug.core.harness.claude_code_session import ChatSession, Content
from aug.core.harness.claude_code_tools import (
    AUG_SERVERS,
    ToolCalls,
    aug_mcp_servers,
    guard_hooks,
    tool_fingerprint,
)
from aug.core.harness.tool_context import ToolContext, pending_approvals
from aug.core.harness.types import (
    Address,
    ApprovalDecision,
    FilePart,
    HarnessEvent,
    Message,
    ModelTier,
    Sender,
    TextPart,
    TurnContext,
    TurnEnded,
    UnsolicitedSink,
)
from aug.core.prompts import CLAUDE_BACKGROUND_TURN, USER_FILE_HINT, USER_IMAGE_HINT
from aug.core.run import AgentRun, run_registry
from aug.core.workspace import session_prompt
from aug.utils.data import DATA_DIR, SKILLS_DIR, WORKSPACE_DIR
from aug.utils.hushed import HushedError, read_secret
from aug.utils.media import MODEL_IMAGE_TYPES
from aug.utils.skills import load_skills

logger = logging.getLogger(__name__)
_cli_logger = logging.getLogger("aug.claude_cli")

CLAUDE_DIR = DATA_DIR / "claude"
# Closed once idle this long: Claude's prompt cache lives an hour, refreshed on each hit.
IDLE_SECONDS = 60 * 60
# Idle processes kept at most; busy ones never count.
MAX_IDLE_PROCESSES = 8
_EVICT_EVERY_SECONDS = 60

# The hushed secret holding the Claude login (``claude setup-token``).
TOKEN_SECRET = "CLAUDE_CODE_OAUTH_TOKEN"
# The plugin name the workspace's skills load under: ``aug:<skill>``.
PLUGIN_NAME = "aug"

_BUNDLED_CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
_CLI_WRAPPER = Path(__file__).parent / "claude_cli.sh"
# What reaches the CLI's environment; everything else of AUG's is cut (claude_cli.sh).
_ENV_KEEP = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "USER",
    "SSL_CERT_FILE",
    "PWD",
    "HUSHED_DIR",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CONFIG_DIR",
    TOKEN_SECRET,
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_AGENT_SDK_VERSION",
    "CLAUDE_AGENT_SDK_CLIENT_APP",
    "CLAUDE_CODE_DISABLE_CRON",
    "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS",
    "CLAUDE_CODE_BG_TASKS_REPORT_RUNNING",
    "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS",
    "DISABLE_AUTOUPDATER",
)
_CLI_ENV = {
    # Scheduling is AUG's (create_task, set_reminder): Claude's cron dies with an
    # evicted process.
    "CLAUDE_CODE_DISABLE_CRON": "1",
    # session_state_changed events: how a turn's end, and a self-started turn, is seen.
    "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1",
    "CLAUDE_CODE_BG_TASKS_REPORT_RUNNING": "0",
    # Wait for every MCP call, so a 30-minute approval wait stays inside its turn.
    "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS": "0",
    # The CLI's version is pinned with the SDK.
    "DISABLE_AUTOUPDATER": "1",
}
# Flag settings: no messages from other Claude sessions (they could carry text between
# chats).
_CLI_SETTINGS = {"crossSessionInbound": "refuse"}

# Set inside a turn the CLI started by itself, which registers no AgentRun.
_UNSOLICITED: ContextVar[bool] = ContextVar("claude_unsolicited", default=False)


@dataclass(frozen=True)
class ClaudeCodeProfile:
    """One Claude Code agent version.

    model / effort: fixed for the profile (a change mid-chat breaks the cache).
    builtin_tools:  Claude's own tools it may use; anything else stays blocked.
    tools:          AUG's tools, the MCP manager's included.
    always_loaded:  names of *tools* Claude has loaded in every request; the rest wait
                    behind its ToolSearch until it looks them up.
    models:         oneshot()'s model per tier.
    """

    model: str
    effort: str
    builtin_tools: Sequence[str]
    tools: Sequence[BaseTool]
    always_loaded: frozenset[str] = frozenset()
    models: Mapping[ModelTier, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if unknown := self.always_loaded - {t.name for t in self.tools}:
            raise ValueError(f"always_loaded names tools the profile lacks: {sorted(unknown)}")


@dataclass
class _Record:
    """What AUG keeps about one chat's Claude session, so it resumes as it started."""

    context_id: str
    session_id: str
    interface: str
    prompt: str
    skills: list[str]
    started: bool = False
    conversation_id: str = ""
    chat_id: str = ""

    @property
    def reply_to(self) -> Address:
        return Address(
            interface=self.interface, conversation_id=self.conversation_id, chat_id=self.chat_id
        )


class ClaudeCodeHarness(Harness):
    def __init__(
        self,
        profile: ClaudeCodeProfile,
        *,
        config_dir: Path = CLAUDE_DIR,
        idle_seconds: float = IDLE_SECONDS,
        max_idle: int = MAX_IDLE_PROCESSES,
    ) -> None:
        self.profile = profile
        self._config_dir = config_dir
        self._records_dir = config_dir / "aug-sessions"
        self._idle_seconds = idle_seconds
        self._max_idle = max_idle
        self._sessions: dict[str, ChatSession] = {}
        self._opening: dict[str, asyncio.Lock] = {}
        self._early: dict[str, list[Content]] = {}  # injected before the session took it
        self._opened: dict[str, AgentRun] = {}
        self._sink: UnsolicitedSink | None = None

    # -- Harness -----------------------------------------------------------------------

    async def _run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        run = run_registry.get(context_id)
        stopped = run.user_requested_stop if run is not None else asyncio.Event()
        try:
            content = await _render(message)
            session = await self._session(context_id, turn.reply_to)
            if stopped.is_set():
                yield TurnEnded(cancelled=True)
                return
            ctx = ToolContext(context_id=context_id, turn=turn, stopped=stopped)
            events = await session.start(content, ctx)
            for early in self._early.pop(context_id, []):
                await session.inject(early)
            async for event in events:
                yield event
        finally:
            self._early.pop(context_id, None)
            await self._mark_started(context_id)

    async def inject(self, message: Message, context_id: str) -> bool:
        if not run_registry.accepting(context_id):
            return False
        content = await _render(message)
        session = self._sessions.get(context_id)
        if session is not None and await session.inject(content):
            return True
        early = self._early.get(context_id)
        if early is None:
            return False
        early.append(content)
        return True

    async def cancel(self, context_id: str) -> bool:
        run = run_registry.get(context_id)
        session = self._sessions.get(context_id)
        if run is None or not run.active:
            # A turn the CLI started by itself registers no run, and stops all the same.
            if session is not None and session.running_unsolicited:
                logger.info("cancel context=%s (self-started turn)", context_id)
                return await session.interrupt()
            return False
        logger.info("cancel context=%s run=%s", context_id, run.id)
        run.request_stop()
        pending_approvals.deny_all(context_id, ApprovalDecision.DENIED)
        if session is not None:
            await session.interrupt()
        return True

    async def compact(self, context_id: str) -> bool:
        record = self._load(context_id)
        if record is None or not record.started:
            return False
        session = await self._session(context_id, record.reply_to)
        ctx = ToolContext(
            context_id=context_id, turn=TurnContext(reply_to=record.reply_to, attended=False)
        )
        ended: TurnEnded | None = None
        events = await session.start([{"type": "text", "text": "/compact"}], ctx, verbatim=False)
        async for event in events:
            if isinstance(event, TurnEnded):
                ended = event
        return ended is not None and ended.error is None

    async def session_prompt(self, context_id: str, interface: str) -> str:
        record = self._load(context_id)
        if record is not None:
            return record.prompt
        return await asyncio.to_thread(_new_prompt, interface)

    async def oneshot(
        self, prompt: str, tier: ModelTier = ModelTier.STRONG, *, system: str = ""
    ) -> str:
        folder = self._config_dir / "oneshot"
        folder.mkdir(parents=True, exist_ok=True)
        options = ClaudeAgentOptions(
            model=self.profile.models.get(tier) or self.profile.model,
            system_prompt=system,
            setting_sources=[],
            tools=[],
            skills=[],
            strict_mcp_config=True,
            permission_mode="dontAsk",
            max_turns=1,
            cwd=str(folder),
            cli_path=str(_CLI_WRAPPER),
            env=await self._env(),
            stderr=_cli_logger.debug,
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"oneshot failed: {message.subtype} {message.errors}")
                return message.result or ""
        raise RuntimeError("oneshot ended without a result")

    @asynccontextmanager
    async def running(self, on_unsolicited: UnsolicitedSink) -> AsyncIterator[None]:
        self._sink = on_unsolicited
        evictor = asyncio.create_task(self._evict_forever(), name="claude-evict")
        try:
            yield
        finally:
            evictor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await evictor
            await asyncio.gather(*(s.close() for s in self._sessions.values()))
            self._sessions.clear()
            self._sink = None

    def _open_turn(self, context_id: str, turn: TurnContext) -> None:
        if _UNSOLICITED.get():
            return
        self._early[context_id] = []  # inject() works from TurnStarted on
        existing = run_registry.get(context_id)
        if existing is not None and existing.active:
            existing.request_stop()  # the run lock should rule this out; never leave two
        run = AgentRun(accepts_injection=turn.attended)
        run_registry.set(context_id, run)
        self._opened[context_id] = run

    def _close_turn(self, context_id: str) -> None:
        if _UNSOLICITED.get():
            return
        self._early.pop(context_id, None)
        run = self._opened.pop(context_id, None)
        if run is None:
            return
        run.active = False
        if run_registry.get(context_id) is run:
            run_registry.pop(context_id)

    # -- sessions ----------------------------------------------------------------------

    async def _session(self, context_id: str, reply_to: Address) -> ChatSession:
        """The chat's warm process: opened, or restarted when its setup changed."""
        lock = self._opening.setdefault(context_id, asyncio.Lock())
        async with lock:
            record = self._load(context_id) or await self._new_record(context_id, reply_to)
            fingerprint = self._fingerprint(record)
            session = self._sessions.get(context_id)
            changed = session is not None and session.fingerprint != fingerprint
            if changed and session.alive and session.has_background_work:
                # Restarting would kill it: the new setup waits for the next turn.
                logger.info("claude setup changed, restart waits for background work")
                changed = False
            if session is not None and (not session.alive or changed):
                if session.alive:
                    logger.info("claude setup changed, restarting context=%s", context_id)
                await session.close()
                self._sessions.pop(context_id, None)
                session = None
            if session is None:
                if record.started and not self._session_file_exists(record.session_id):
                    logger.warning(
                        "claude session file gone, starting afresh context=%s", context_id
                    )
                    record.session_id, record.started = str(uuid.uuid4()), False
                    self._save(record)
                calls = ToolCalls()
                session = ChatSession(
                    context_id,
                    await self._options(record, calls),
                    calls,
                    fingerprint=fingerprint,
                    reply_to=reply_to,
                    on_unsolicited=lambda events: self._unsolicited(context_id, events),
                )
                await session.open()
                self._sessions[context_id] = session
            session.reply_to = reply_to
            return session

    def _session_file_exists(self, session_id: str) -> bool:
        """Whether Claude still has *session_id* on disk (a wiped config dir has not)."""
        return any((self._config_dir / "projects").glob(f"*/{session_id}.jsonl"))

    async def _options(self, record: _Record, calls: ToolCalls) -> ClaudeAgentOptions:
        plugin = self._plugin_dir()
        builtin = list(self.profile.builtin_tools)
        return ClaudeAgentOptions(
            model=self.profile.model,
            effort=self.profile.effort,  # type: ignore[arg-type]
            system_prompt={"type": "preset", "preset": "claude_code", "append": record.prompt},
            setting_sources=[],
            tools=builtin,
            allowed_tools=[*builtin, *(f"mcp__{server}" for server in AUG_SERVERS)],
            permission_mode="dontAsk",
            mcp_servers=aug_mcp_servers(self.profile.tools, self.profile.always_loaded, calls),
            strict_mcp_config=True,
            plugins=[{"type": "local", "path": str(plugin)}],
            skills=list(record.skills),
            hooks=guard_hooks(),  # type: ignore[arg-type]
            include_partial_messages=True,
            settings=json.dumps(_CLI_SETTINGS),
            cwd=str(WORKSPACE_DIR),
            cli_path=str(_CLI_WRAPPER),
            env=await self._env(),
            stderr=_cli_logger.debug,
            resume=record.session_id if record.started else None,
            session_id=None if record.started else record.session_id,
        )

    def _fingerprint(self, record: _Record) -> str:
        setup = {
            "prompt": record.prompt,
            "skills": record.skills,
            "model": self.profile.model,
            "effort": self.profile.effort,
            "builtin_tools": list(self.profile.builtin_tools),
            "tools": tool_fingerprint(self.profile.tools),
            "always_loaded": sorted(self.profile.always_loaded),
            "env": _CLI_ENV,
            "settings": _CLI_SETTINGS,
        }
        return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()

    async def _env(self) -> dict[str, str]:
        env = {
            **_CLI_ENV,
            "CLAUDE_CONFIG_DIR": str(self._config_dir),
            "AUG_CLAUDE_CLI": str(_BUNDLED_CLI),
            "AUG_CLAUDE_ENV_KEEP": " ".join(_ENV_KEEP),
        }
        try:
            env[TOKEN_SECRET] = await asyncio.to_thread(read_secret, TOKEN_SECRET)
        except HushedError:
            logger.debug("no %s in hushed: the CLI uses its own login", TOKEN_SECRET)
        return env

    async def _unsolicited(self, context_id: str, events: AsyncIterator[HarnessEvent]) -> None:
        """Deliver a turn the CLI started by itself, logged like any other.  One that
        ended before anything happened (a user's turn took it over) is not sent."""
        first = await anext(events)
        if isinstance(first, TurnEnded):
            return
        events = _prepend(first, events)
        session = self._sessions.get(context_id)
        if session is None or self._sink is None:
            async for _ in events:
                pass
            logger.warning("claude turn dropped context=%s: nowhere to deliver it", context_id)
            return
        reply_to = session.reply_to
        message = Message(
            parts=[TextPart(text=CLAUDE_BACKGROUND_TURN)],
            sender=Sender(interface=reply_to.interface, user_id=""),
        )
        _UNSOLICITED.set(True)
        turn = TurnContext(reply_to=reply_to, attended=False)
        await self._sink(reply_to, context_id, self._turn(message, context_id, turn, events))

    async def _evict_forever(self) -> None:
        while True:
            await asyncio.sleep(_EVICT_EVERY_SECONDS)
            try:
                await self._evict()
            except Exception:
                logger.exception("claude eviction failed")

    async def _evict(self) -> None:
        """Close idle processes past IDLE_SECONDS, and the oldest past MAX_IDLE_PROCESSES."""
        now = time.monotonic()
        idle = sorted(
            ((cid, s) for cid, s in self._sessions.items() if not s.busy),
            key=lambda item: item[1].last_used,
        )
        over = max(0, len(idle) - self._max_idle)
        for index, (context_id, session) in enumerate(idle):
            due = index < over or now - session.last_used >= self._idle_seconds
            # Checked again: a turn may have started while an earlier one was closing.
            if (due or not session.alive) and not session.busy:
                self._sessions.pop(context_id, None)
                await session.close()

    def _plugin_dir(self) -> Path:
        """The folder the CLI loads as plugin PLUGIN_NAME: its manifest and a link to the
        workspace's skills, nothing else, so no hooks or commands the agent writes into
        the workspace ever load."""
        plugin = self._config_dir / "plugin"
        manifest = plugin / ".claude-plugin" / "plugin.json"
        if not manifest.exists():
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"name": PLUGIN_NAME}) + "\n")
        skills = plugin / "skills"
        if not skills.is_symlink() or skills.readlink() != SKILLS_DIR:
            skills.unlink(missing_ok=True)
            skills.symlink_to(SKILLS_DIR, target_is_directory=True)
        return plugin

    # -- records -----------------------------------------------------------------------

    async def _new_record(self, context_id: str, reply_to: Address) -> _Record:
        prompt = await asyncio.to_thread(_new_prompt, reply_to.interface)
        skills = [
            f"{PLUGIN_NAME}:{s.name}" for s in await asyncio.to_thread(load_skills, SKILLS_DIR)
        ]
        record = _Record(
            context_id=context_id,
            session_id=str(uuid.uuid4()),
            interface=reply_to.interface,
            prompt=prompt,
            skills=skills,
            conversation_id=reply_to.conversation_id,
            chat_id=reply_to.chat_id,
        )
        self._save(record)
        return record

    async def _mark_started(self, context_id: str) -> None:
        record = self._load(context_id)
        session = self._sessions.get(context_id)
        if record is None or session is None or session.session_id is None:
            return
        if not record.started or record.session_id != session.session_id:
            record.started = True
            record.session_id = session.session_id
            self._save(record)

    def _record_path(self, context_id: str) -> Path:
        return self._records_dir / f"{hashlib.sha256(context_id.encode()).hexdigest()[:32]}.json"

    def _load(self, context_id: str) -> _Record | None:
        try:
            return _Record(**json.loads(self._record_path(context_id).read_text()))
        except FileNotFoundError:
            return None
        except (ValueError, TypeError):
            logger.warning("unreadable claude session record context=%s", context_id)
            return None

    def _save(self, record: _Record) -> None:
        path = self._record_path(record.context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(asdict(record)))
        partial.replace(path)


def _new_prompt(interface: str) -> str:
    return session_prompt(interface, skills_index=False)


async def _render(message: Message) -> Content:
    """*message* as Claude content blocks: the time first, images inline with their
    path, other files as a hint with their path."""
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    blocks: Content = [{"type": "text", "text": f"[{now}]"}]
    for part in message.parts:
        match part:
            case TextPart(text=text) if text:
                blocks.append({"type": "text", "text": text})
            case FilePart(path=path, media_type=media_type) if media_type in MODEL_IMAGE_TYPES:
                try:
                    data = await asyncio.to_thread(path.read_bytes)
                except OSError:
                    logger.warning("image unreadable: %s", path)
                else:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(data).decode(),
                            },
                        }
                    )
                blocks.append({"type": "text", "text": USER_IMAGE_HINT.format(path=path)})
            case FilePart(path=path, media_type=media_type, filename=filename):
                blocks.append(
                    {
                        "type": "text",
                        "text": USER_FILE_HINT.format(
                            filename=filename, media_type=media_type, path=path
                        ),
                    }
                )
    return blocks


async def _prepend(
    first: HarnessEvent, rest: AsyncIterator[HarnessEvent]
) -> AsyncIterator[HarnessEvent]:
    yield first
    async for event in rest:
        yield event
