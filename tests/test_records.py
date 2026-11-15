"""RecordStore: a chat's session record, as JSON, read back as it was written."""

import json
from dataclasses import dataclass, replace

from aug.core.harness.records import RecordStore, SessionRecord
from aug.core.harness.types import ContextUsage


@dataclass(frozen=True, kw_only=True)
class _Record(SessionRecord):
    session_id: str
    started: bool = False


def _store(tmp_path) -> RecordStore[_Record]:
    return RecordStore(tmp_path / "records", _Record)


async def test_a_record_reads_back_as_written(tmp_path):
    store = _store(tmp_path)
    context = ContextUsage(used=10, window=100, breakdown=(("messages", 10),))
    record = _Record(
        context_id="c", interface="telegram", prompt="p", session_id="s", context=context
    )
    await store.save(record)
    assert await store.load("c") == record
    assert (await store.load("c")).reply_to.interface == "telegram"
    assert await store.load("other") is None
    assert not list((tmp_path / "records").glob("*.partial"))


async def test_a_stored_context_reads_back_from_its_json(tmp_path):
    store = _store(tmp_path)
    await store.save(_Record(context_id="c", interface="i", prompt="p", session_id="s"))
    (path,) = (tmp_path / "records").iterdir()
    raw = json.loads(path.read_text())
    raw["context"] = {"used": 5, "window": 50, "breakdown": [["messages", 5]]}
    raw["dropped_field"] = True  # a field this record type does not have
    path.write_text(json.dumps(raw))

    record = await store.load("c")
    assert record.context == ContextUsage(used=5, window=50, breakdown=(("messages", 5),))


async def test_an_unreadable_record_is_none(tmp_path):
    store = _store(tmp_path)
    await store.save(_Record(context_id="c", interface="i", prompt="p", session_id="s"))
    (path,) = (tmp_path / "records").iterdir()
    path.write_text("{not json")
    assert await store.load("c") is None
    path.write_text(json.dumps({"context_id": "c"}))  # fields missing
    assert await store.load("c") is None


async def test_a_stored_context_that_no_longer_reads_is_dropped(tmp_path):
    store = _store(tmp_path)
    await store.save(_Record(context_id="c", interface="i", prompt="p", session_id="s"))
    (path,) = (tmp_path / "records").iterdir()
    raw = json.loads(path.read_text())
    raw["context"] = {"used": "lots"}
    path.write_text(json.dumps(raw))
    record = await store.load("c")
    assert record is not None and record.context is None


async def test_an_update_applies_to_the_stored_record(tmp_path):
    store = _store(tmp_path)
    assert await store.update("c", lambda r: r) is None  # no record: nothing saved
    await store.save(_Record(context_id="c", interface="i", prompt="p", session_id="s"))
    updated = await store.update("c", lambda r: replace(r, started=True))
    assert updated == await store.load("c") and updated.started
