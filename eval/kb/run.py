"""Knowledge-base eval harness — drives the real AUG over /chat/invoke.

    uv run python -m eval.kb.run --agent v13_kimi [--runs 3] [--only-questions OWNER]

1. Writes: every event goes to a fresh chat as a fake person would send it (files
   through /files/upload + attachments), then everything it wrote is backdated to the
   event's date, and the KB database is checked for what should now be there.
2. Questions: each asked in a brand-new chat, ``--runs`` times.  Graded by code (tool
   trace from the checkpointer, evidence in what the KB returned, must / must-not
   strings in the answer, the right file sent back) and by an LLM judge on a
   different model from the one under test.
3. Ops: kb_tool latencies from the container log.

Writes a Markdown report and the raw JSON to eval-results/kb/<run>/.
Needs the eval stack (eval/kb/docker-compose.yml) up and the LiteLLM proxy reachable.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import asyncpg
import httpx
import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from aug.core.llm import build_chat_model
from eval.kb.fixtures import IMAGES, SCANS, TEXT_PDFS
from eval.kb.fixtures import build as build_fixtures

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval-results" / "kb"
AUG_URL = "http://localhost:8013"
CHECKPOINT_DSN = "postgresql://aug:password@localhost:5434/aug"
KB_DSN = "postgresql://aug:password@localhost:5435/aug_kb"
CONTAINER = "aug-kb-eval-aug-1"
KB_READ_TOOLS = {"kb_query", "kb_get"}
CONCURRENCY = 4
FIXTURE_TEXT = {**TEXT_PDFS, **SCANS, **IMAGES}


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: str


@dataclass
class Turn:
    thread_id: str
    response: str
    seconds: float
    tools: list[ToolCall]
    error: str = ""


@dataclass
class EventResult:
    id: str
    kind: str
    turn: Turn
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@dataclass
class QuestionRun:
    turn: Turn
    searched: bool
    evidence_found: bool
    must_ok: bool
    must_not_ok: bool
    file_ok: bool | None
    verdict: str
    reason: str
    passed: bool
    cause: str


@dataclass
class QuestionResult:
    id: str
    cat: str
    ask: str
    runs: list[QuestionRun]

    @property
    def passed(self) -> bool:
        return sum(r.passed for r in self.runs) * 2 > len(self.runs)


class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "wrong", "made_up"]
    reason: str


# ── main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    args = _parse_args()
    _load_env()
    dataset = yaml.safe_load((ROOT / "eval/kb/dataset.yaml").read_text())
    if args.only_events:
        ids = set(args.only_events.split(","))
        dataset["events"] = [e for e in dataset["events"] if e["id"] in ids]
    if args.only_ask:
        ids = set(args.only_ask.split(","))
        dataset["questions"] = [q for q in dataset["questions"] if q["id"] in ids]

    run_id = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    owner = args.owner or f"eval-{run_id}"
    out = RESULTS / f"{run_id}-{args.agent}"
    out.mkdir(parents=True, exist_ok=True)
    fixtures = build_fixtures(RESULTS / "fixtures")
    log_since = datetime.now(tz=UTC).isoformat()
    print(f"run {run_id}  agent={args.agent}  owner={owner}  → {out}")

    async with (
        httpx.AsyncClient(base_url=AUG_URL, timeout=300) as http,
        AsyncPostgresSaver.from_conn_string(CHECKPOINT_DSN, serde=_serde()) as saver,
    ):
        kb = await asyncpg.connect(KB_DSN)
        api = _Api(http, saver, owner, args.agent)

        events: list[EventResult] = []
        if not args.owner:
            for event in dataset["events"]:
                result = await _run_event(api, kb, owner, event, fixtures)
                events.append(result)
                mark = "ok " if result.passed else "FAIL"
                print(
                    f"  [{mark}] {event['id']:<4} {event['kind']:<10} {result.turn.seconds:5.1f}s "
                    f"{' '.join(k for k, v in result.checks.items() if not v)}"
                )

        judge_model = "gpt-4.1" if "claude" in args.agent else "claude-sonnet-4-6"
        judge = _judge(judge_model, _ground_truth(dataset))
        sem = asyncio.Semaphore(CONCURRENCY)
        db_text = await _all_live_text(kb, owner)

        async def ask(question: dict) -> QuestionResult:
            runs = []
            for n in range(args.runs):
                async with sem:
                    turn = await api.chat(f"{owner}-{question['id']}-r{n}", question["ask"])
                runs.append(await _grade(question, turn, fixtures, judge, db_text))
            result = QuestionResult(question["id"], question["cat"], question["ask"], runs)
            mark = "ok " if result.passed else "FAIL"
            causes = ",".join(sorted({r.cause for r in runs if not r.passed}))
            print(f"  [{mark}] {question['id']} {question['cat']:<15} {causes}")
            return result

        questions = await asyncio.gather(*(ask(q) for q in dataset["questions"]))
        await kb.close()

    ops = _ops_from_logs(log_since, owner)
    report = _report(args.agent, owner, judge_model, events, list(questions), ops)
    (out / "report.md").write_text(report)
    (out / "raw.json").write_text(
        json.dumps(
            {
                "events": [asdict(e) for e in events],
                "questions": [asdict(q) for q in questions],
                "ops": ops,
            },
            default=str,
            indent=1,
            ensure_ascii=False,
        )
    )
    print("\n" + report.split("\n## Failures")[0])
    print(f"\nreport: {out / 'report.md'}")


# ── talking to AUG ───────────────────────────────────────────────────────────


class _Api:
    def __init__(self, http: httpx.AsyncClient, saver, owner: str, agent: str) -> None:
        self._http = http
        self._saver = saver
        self._owner = owner
        self._agent = agent
        self._headers = {"X-API-Key": os.environ["API_KEY"]}

    async def upload(self, path: Path) -> str:
        r = await self._http.post(
            "/files/upload", headers=self._headers, files={"file": (path.name, path.read_bytes())}
        )
        r.raise_for_status()
        return r.json()["file_id"]

    async def chat(self, thread_id: str, message: str, attachments: list | None = None) -> Turn:
        started = time.monotonic()
        body = {
            "thread_id": thread_id,
            "message": message,
            "agent": self._agent,
            "user_id": self._owner,
            "attachments": attachments or [],
        }
        try:
            r = await self._http.post("/chat/invoke", headers=self._headers, json=body)
            r.raise_for_status()
            text, error = r.json()["response"], ""
        except Exception as exc:
            text, error = "", f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - started
        return Turn(thread_id, text, seconds, await self._trace(thread_id), error)

    async def _trace(self, thread_id: str) -> list[ToolCall]:
        snapshot = await self._saver.aget_tuple({"configurable": {"thread_id": thread_id}})
        if snapshot is None:
            return []
        messages = snapshot.checkpoint["channel_values"].get("messages", [])
        results = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
        calls = []
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    calls.append(ToolCall(tc["name"], tc["args"], results.get(tc["id"], "")))
        return calls


# ── writes ───────────────────────────────────────────────────────────────────


async def _run_event(
    api: _Api, kb: asyncpg.Connection, owner: str, event: dict, fixtures: dict[str, Path]
) -> EventResult:
    started_at = await kb.fetchval("SELECT now()")
    objects_before = await _count_objects(kb, owner)
    attachments = []
    if event.get("file"):
        attachments = [{"file_id": await api.upload(fixtures[event["file"]])}]
    turn = await api.chat(f"{owner}-{event['id']}", event["message"], attachments)
    result = EventResult(event["id"], event["kind"], turn)
    writes = [t for t in turn.tools if t.name == "kb_write"]
    # A fact about a node that already exists is saved with kb_relate — just as valid.
    saves = [t for t in turn.tools if t.name in ("kb_write", "kb_relate")]
    should_save = (
        event.get("save", True) and event["kind"] not in ("forget",) and not event.get("ask")
    )

    result.checks["no_error"] = not turn.error
    if event["kind"] == "forget":
        result.checks["forget_called"] = any(t.name == "kb_forget" for t in turn.tools)
    elif event.get("ask"):
        pass  # graded by asked_which below; saving before the answer is optional
    elif should_save:
        result.checks["saved"] = bool(saves)
        result.checks["save_ok"] = any(
            '"status": "ok"' in w.result or '"fact_id"' in w.result for w in saves
        )
    else:
        result.checks["not_saved"] = await _count_objects(kb, owner) == objects_before

    if event.get("ask"):
        result.checks["asked_which"] = (
            any('"ask": ""' not in w.result for w in writes) or "which" in turn.response.lower()
        )

    facts_live = await _facts(kb, owner, closed=False)
    facts_closed = await _facts(kb, owner, closed=True)
    text_live = await _live_text(kb, owner)
    for i, expected in enumerate(event.get("facts") or []):
        result.checks[f"fact[{i}]"] = _contains(facts_live, expected)
    for i, expected in enumerate(event.get("closed") or []):
        result.checks[f"closed[{i}]"] = _contains(facts_closed, expected)
    for i, expected in enumerate(event.get("stored") or []):
        result.checks[f"stored[{i}]"] = _contains(text_live, expected)
    for i, gone in enumerate(event.get("forget") or []):
        result.checks[f"forgotten[{i}]"] = not _contains(text_live + facts_live, gone)

    await _backdate(kb, owner, started_at, event["at"])
    return result


async def _backdate(kb: asyncpg.Connection, owner: str, since: datetime, day) -> None:
    """Move what an event wrote to the event's day.

    The agent and the extractor saw the real clock, so "today" inside the text they
    wrote is the run date — rewritten to the event day as well, or a later answer
    would truthfully quote the wrong date.  (Chunk embeddings keep the old date;
    one token, negligible.)
    """
    at = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
    async with kb.transaction():
        await kb.execute("SET LOCAL ROLE kb_session")  # superusers bypass RLS
        await kb.execute("SELECT set_config('app.owner_id', $1, true)", owner)
        real = since.date()
        if real != day:
            for old, new in zip(_date_spellings(real), _date_spellings(day), strict=True):
                for sql in (
                    """UPDATE kb_object SET title = replace(title, $2, $3),
                           summary = replace(summary, $2, $3),
                           knowledge = replace(knowledge, $2, $3),
                           derived_text = replace(derived_text, $2, $3)
                       WHERE received_at >= $1""",
                    """UPDATE kb_chunk SET context = replace(context, $2, $3),
                           text = replace(text, $2, $3)
                       WHERE node_id IN (SELECT id FROM kb_object WHERE received_at >= $1)""",
                    # Profiles rebuilt during this event: only its own facts still
                    # carry the real date — earlier events' were rewritten already.
                    """UPDATE kb_chunk SET text = replace(text, $2, $3)
                       WHERE is_profile AND node_id IN (
                           SELECT subject_id FROM kb_fact WHERE recorded_at >= $1
                           UNION SELECT value_node_id FROM kb_fact WHERE recorded_at >= $1)""",
                    """UPDATE kb_fact SET value_literal = replace(value_literal, $2, $3)
                       WHERE recorded_at >= $1 AND value_literal IS NOT NULL""",
                ):
                    await kb.execute(sql, since, old, new)
            await kb.execute(
                "UPDATE kb_object SET dated_at = $3 WHERE received_at >= $1 AND dated_at = $2",
                since,
                real,
                day,
            )
        for sql in (
            "UPDATE kb_object SET received_at = $2 WHERE received_at >= $1",
            "UPDATE kb_object SET deleted_at = $2 WHERE deleted_at >= $1",
            "UPDATE kb_entity SET created_at = $2 WHERE created_at >= $1",
            "UPDATE kb_entity SET deleted_at = $2 WHERE deleted_at >= $1",
            "UPDATE kb_predicate SET created_at = $2 WHERE created_at >= $1",
            "UPDATE kb_fact SET valid_from = $2, recorded_at = $2 WHERE recorded_at >= $1",
            "UPDATE kb_fact SET valid_to = $2 WHERE valid_to >= $1",
            "UPDATE kb_fact SET deleted_at = $2 WHERE deleted_at >= $1",
        ):
            await kb.execute(sql, since, at)


def _date_spellings(d) -> list[str]:
    return [d.isoformat(), f"{d.day} {d:%B %Y}", f"{d:%B} {d.day}, {d.year}"]


async def _owned(kb: asyncpg.Connection, owner: str, sql: str) -> list[asyncpg.Record]:
    async with kb.transaction():
        await kb.execute("SET LOCAL ROLE kb_session")  # superusers bypass RLS
        await kb.execute("SELECT set_config('app.owner_id', $1, true)", owner)
        return await kb.fetch(sql)


async def _count_objects(kb: asyncpg.Connection, owner: str) -> int:
    return len(await _owned(kb, owner, "SELECT id FROM kb_object"))


async def _facts(kb: asyncpg.Connection, owner: str, *, closed: bool) -> str:
    rows = await _owned(
        kb,
        owner,
        f"""
        SELECT f.subject_id || ' ' || f.predicate || ' ' ||
               coalesce(f.value_literal, e.display_name, o.title, f.value_node_id) AS line
        FROM kb_fact f
        LEFT JOIN kb_entity e ON e.id = f.value_node_id
        LEFT JOIN kb_object o ON o.id = f.value_node_id
        WHERE f.deleted_at IS NULL AND f.valid_to IS {"NOT" if closed else ""} NULL""",
    )
    return "\n".join(r["line"] for r in rows)


async def _live_text(kb: asyncpg.Connection, owner: str) -> str:
    rows = await _owned(
        kb,
        owner,
        """
        SELECT c.context || ' ' || c.text AS t FROM kb_chunk c
        JOIN kb_object o ON o.id = c.node_id AND o.deleted_at IS NULL""",
    )
    return "\n".join(r["t"] for r in rows)


async def _all_live_text(kb: asyncpg.Connection, owner: str) -> str:
    return await _live_text(kb, owner) + "\n" + await _facts(kb, owner, closed=False)


# ── questions ────────────────────────────────────────────────────────────────


async def _grade(
    question: dict, turn: Turn, fixtures: dict[str, Path], judge, db_text: str
) -> QuestionRun:
    reads = [t for t in turn.tools if t.name in KB_READ_TOOLS]
    searched = bool(reads)
    returned = "\n".join(t.result for t in reads)
    must = question.get("must") or []
    must_not = question.get("must_not") or []
    evidence = question.get("evidence") or (must[0] if must else "")
    evidence_found = _contains(returned, evidence) if evidence else True
    must_ok = all(_contains(turn.response, m) for m in must)
    must_not_ok = not any(_contains(turn.response, m) for m in must_not)
    file_ok = None
    if question.get("file"):
        sha = hashlib.sha256(fixtures[question["file"]].read_bytes()).hexdigest()
        file_ok = any(
            t.name == "respond_with_file" and sha in json.dumps(t.args) for t in turn.tools
        )

    if question["cat"] == "file":
        verdict, reason = ("correct", "file sent") if file_ok else ("wrong", "file not sent")
    elif turn.error:
        verdict, reason = "wrong", turn.error
    else:
        verdict, reason = await judge(question, turn.response)

    passed = (
        not turn.error and must_ok and must_not_ok and file_ok is not False and verdict == "correct"
    )
    cause = ""
    if not passed:
        if turn.error:
            cause = "error"
        elif verdict == "made_up":
            cause = "made_up"
        elif not searched:
            cause = "agent_didnt_look"
        elif evidence and not evidence_found:
            cause = "search_miss" if _contains(db_text, evidence) else "extraction"
        elif not must_not_ok:
            cause = "leaked"
        else:
            cause = "agent_misread"
    return QuestionRun(
        turn,
        searched,
        evidence_found,
        must_ok,
        must_not_ok,
        file_ok,
        verdict,
        reason,
        passed,
        cause,
    )


def _ground_truth(dataset: dict) -> str:
    """Everything the fake person ever told the assistant, documents included — what
    the judge needs to tell a true extra detail from an invented one."""
    lines = [f"About the user: {dataset['persona']}"]
    for event in dataset["events"]:
        line = f"[{event['at']}] user said: {event['message']}"
        if event.get("ask"):
            # The harness never answers the "which one?", so this was never stored —
            # "nothing on file" is a correct answer about it.
            line += "  (ambiguous; the user never said which one, so it was not saved)"
        lines.append(line)
        if event.get("file"):
            lines.append(
                f"    and sent {event['file']}: " + " | ".join(FIXTURE_TEXT[event["file"]])
            )
    return "\n".join(lines)


def _judge(model: str, ground_truth: str):
    llm = build_chat_model(model, temperature=0.0, timeout=120).with_structured_output(
        Verdict, method="function_calling"
    )

    async def judge(question: dict, answer: str) -> tuple[str, str]:
        expected = question.get("judge") or (
            "answers with: "
            + "; ".join(
                " / ".join(m) if isinstance(m, list) else m for m in question.get("must", [])
            )
        )
        prompt = (
            "You grade an assistant that answers from its own knowledge base.\n\n"
            f"GROUND TRUTH — everything the user ever told it, in order (later messages "
            f"override earlier ones; anything they asked to forget must not come back):\n"
            f"{ground_truth}\n\n"
            f"Question: {question['ask']}\n"
            f"A correct answer {expected}.\n"
            f"Must NOT contain: {question.get('must_not') or 'n/a'}\n\n"
            f"Assistant's answer:\n{answer}\n\n"
            "verdict: correct (gives the expected information; extra detail is fine), "
            "partial (some of it, or hedged when it shouldn't be), wrong (missing or "
            "incorrect, outdated, or says it doesn't know when it should), made_up (states "
            "a specific value — number, code, date, address, name — that is NOT supported by "
            "the ground truth). Extra details that ARE in the ground truth are fine. Give a "
            "one-sentence reason."
        )
        try:
            v = await llm.ainvoke(prompt)
            return v.verdict, v.reason
        except Exception as exc:
            return "wrong", f"judge failed: {exc}"

    return judge


# ── ops ──────────────────────────────────────────────────────────────────────


def _ops_from_logs(since: str, owner: str) -> dict[str, Any]:
    logs = subprocess.run(
        ["docker", "logs", "--since", since, CONTAINER], capture_output=True, text=True
    ).stdout
    lines = [ln for ln in (logs.splitlines()) if "kb_tool" in ln and f"owner={owner}" in ln]
    ms: dict[str, list[int]] = {}
    statuses: dict[str, int] = {}
    for line in lines:
        m = re.search(r"tool=(\w+) owner=\S+ status=(\w+) ms=(\d+)(.*)", line)
        if not m:
            continue
        tool, status, millis, rest = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        key = f"{tool}:{status}"
        statuses[key] = statuses.get(key, 0) + 1
        bucket = tool
        if tool == "kb_write":
            bucket = "kb_write_file" if "file=True" in rest else "kb_write_text"
        ms.setdefault(bucket, []).append(millis)
    errors = [ln for ln in logs.splitlines() if "[ERROR]" in ln or "Traceback" in ln]
    return {
        "latency_ms": {
            k: {"n": len(v), "p50": _pct(v, 50), "p95": _pct(v, 95), "max": max(v)}
            for k, v in ms.items()
        },
        "statuses": statuses,
        "error_lines": errors[:30],
    }


def _pct(values: list[int], p: int) -> int:
    if len(values) == 1:
        return values[0]
    return int(statistics.quantiles(values, n=100, method="inclusive")[p - 1])


# ── report ───────────────────────────────────────────────────────────────────


def _report(
    agent: str,
    owner: str,
    judge_model: str,
    events: list[EventResult],
    questions: list[QuestionResult],
    ops: dict,
) -> str:
    def rate(items) -> str:
        items = list(items)
        return f"{sum(items)}/{len(items)}" if items else "—"

    cats: dict[str, list[QuestionResult]] = {}
    for q in questions:
        cats.setdefault(q.cat, []).append(q)
    never = cats.get("never_told", [])
    made_up = sum(1 for q in never for r in q.runs if r.verdict == "made_up")
    forgotten_leaks = sum(1 for q in cats.get("forgotten", []) for r in q.runs if not r.must_not_ok)
    silent = sum(
        1
        for e in events
        if e.checks.get("save_ok")
        and not all(v for k, v in e.checks.items() if k.startswith(("fact", "stored")))
    )
    chit = [e for e in events if e.kind == "no_save"]
    chit_saved = sum(1 for e in chit if not e.checks.get("not_saved", True))
    all_runs = [r for q in questions for r in q.runs]
    needs_search = [r for q in questions if q.cat not in ("ambiguous",) for r in q.runs]
    exact = cats.get("direct", []) + cats.get("exact_token", [])
    lat = ops["latency_ms"]

    lines = [
        f"# KB eval — {agent}",
        "",
        f"owner `{owner}` · judge `{judge_model}` · {len(events)} writes · "
        f"{len(questions)} questions × {len(questions[0].runs) if questions else 0} runs",
        "",
        "## Pass bar",
        "",
        "| Measure | Result | Target |",
        "| --- | --- | --- |",
        f"| Made-up answers on never-told | {made_up} | 0 |",
        f"| Forgotten items resurfacing | {forgotten_leaks} | 0 |",
        f"| Writes reported ok that saved nothing expected | {silent} | 0 |",
        f"| Direct + exact-token questions | {rate(q.passed for q in exact)} | 100% |",
        f"| All questions | {rate(q.passed for q in questions)} | ≥ 90% |",
        f"| Searched when needed (runs) | {rate(r.searched for r in needs_search)} | ≥ 95% |",
        f"| Evidence in what the KB returned (runs, searched) | "
        f"{rate(r.evidence_found for r in all_runs if r.searched)} | ≥ 90% |",
        f"| Write events fully passing | {rate(e.passed for e in events)} | — |",
        f"| Chit-chat wrongly saved | {chit_saved}/{len(chit)} | ≤ 1/6 |",
        f"| kb_query p95 | {lat.get('kb_query', {}).get('p95', '—')} ms | < 1000 ms |",
        f"| kb_write text-only p95 | {lat.get('kb_write_text', {}).get('p95', '—')} ms | "
        "< 5000 ms |",
        f"| kb_write with file p95 | {lat.get('kb_write_file', {}).get('p95', '—')} ms | — |",
        "",
        "## By category",
        "",
        "| Category | Passed |",
        "| --- | --- |",
        *[f"| {c} | {rate(q.passed for q in qs)} |" for c, qs in cats.items()],
        "",
        "## Failures",
        "",
    ]
    for e in events:
        if not e.passed:
            failed = ", ".join(k for k, v in e.checks.items() if not v)
            lines.append(
                f"- **{e.id}** ({e.kind}) failed {failed} — thread `{e.turn.thread_id}` "
                f"— tools {[t.name for t in e.turn.tools]} — reply: "
                f"{e.turn.response[:200]!r}"
            )
    for q in questions:
        for r in q.runs:
            if not r.passed:
                lines.append(
                    f"- **{q.id}** ({q.cat}) `{r.cause}` — {q.ask!r} → {r.turn.response[:240]!r} "
                    f"— judge: {r.verdict} ({r.reason}) — tools "
                    f"{[t.name for t in r.turn.tools]} — thread `{r.turn.thread_id}`"
                )
    lines += ["", "## Ops", "", "```", json.dumps(ops, indent=1)[:4000], "```"]
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def _contains(haystack: str, needle: str | list[str]) -> bool:
    options = needle if isinstance(needle, list) else [needle]
    hay = _norm(haystack)
    return any(_norm(o) in hay for o in options)


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("aug.core.tools.approval", "ApprovalRequest"),
            ("aug.core.tools.approval", "ApprovalDecision"),
        ]
    )


def _load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("DATABASE_URL", CHECKPOINT_DSN)
    os.environ["DEBUG"] = "true"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agent", default="v13_kimi")
    p.add_argument("--runs", type=int, default=1, help="times each question is asked")
    p.add_argument("--owner", help="reuse an existing owner: skip writes, ask questions only")
    p.add_argument("--only-events", help="comma-separated event ids")
    p.add_argument("--only-ask", help="comma-separated question ids")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
