"""The workspace: layout, the one-time migration, and the session prompt built from it."""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from aug.core.agents.chat_agent import AugAgent
from aug.core.harness import Address, Message, Sender, TextPart, TurnContext, TurnEnded
from aug.core.harness.langgraph import LangGraphHarness
from aug.core.memory import (
    MEMORY_INDEX_MAX_CHARS,
    append_note,
    run_deep_consolidation,
    run_light_consolidation,
)
from aug.core.tools.skills import get_skill
from aug.core.workspace import AGENTS_TEMPLATE, init_workspace, session_prompt
from aug.utils.skills import load_skill, write_skill_md

TURN = TurnContext(reply_to=Address(interface="telegram", conversation_id="c"), attended=True)


def _legacy(workspace: Path) -> Path:
    return workspace.parent / "legacy"


def _skill(folder: Path, name: str, description: str = "Does a thing") -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody of {name}.\n"
    )


# ---------------------------------------------------------------------------
# init_workspace
# ---------------------------------------------------------------------------


def test_a_fresh_install_gets_the_template_and_an_index(workspace: Path) -> None:
    init_workspace()

    assert (workspace / "AGENTS.md").read_text() == AGENTS_TEMPLATE.read_text()
    index = (workspace / "memory" / "MEMORY.md").read_text()
    assert "- user.md: " in index
    assert "- context.md: " in index
    assert (workspace / "skills").is_dir()


def test_the_old_memory_and_skills_are_copied_in(workspace: Path) -> None:
    legacy = _legacy(workspace)
    (legacy / "memory").mkdir(parents=True)
    (legacy / "memory" / "user.md").write_text("Vadim lives in London.")
    (legacy / "memory" / "reflections-archive-2026-08.md").write_text("old")
    _skill(legacy / "skills", "homelab")

    init_workspace()

    assert (workspace / "memory" / "user.md").read_text() == "Vadim lives in London."
    assert (workspace / "memory" / "reflections-archive-2026-08.md").exists()
    assert (workspace / "skills" / "homelab" / "SKILL.md").exists()
    assert "- reflections-archive-2026-08.md" in (workspace / "memory" / "MEMORY.md").read_text()
    assert (legacy / "memory" / "user.md").exists()  # the old folders stay


def test_an_old_memory_md_is_not_taken_for_the_index(workspace: Path) -> None:
    (_legacy(workspace) / "memory").mkdir(parents=True)
    (_legacy(workspace) / "memory" / "memory.md").write_text("## Patterns\nlong unused")

    init_workspace()

    names = [p.name for p in (workspace / "memory").iterdir()]
    assert "memory.md" not in names
    assert "long unused" not in (workspace / "memory" / "MEMORY.md").read_text()


def test_the_old_self_md_becomes_the_persona(workspace: Path) -> None:
    (_legacy(workspace) / "memory").mkdir(parents=True)
    (_legacy(workspace) / "memory" / "self.md").write_text("I am AUG, Vadim's druzhban.\n")

    init_workspace()

    agents = (workspace / "AGENTS.md").read_text()
    assert agents.startswith("# AUG\n\nI am AUG, Vadim's druzhban.\n\n## Replies")
    assert "brilliant friend" not in agents  # the template's persona is replaced
    assert "An empty reply is fine" in agents


def test_running_it_again_overwrites_nothing(workspace: Path) -> None:
    legacy = _legacy(workspace)
    (legacy / "memory").mkdir(parents=True)
    (legacy / "memory" / "user.md").write_text("old fact")
    _skill(legacy / "skills", "homelab", "old description")
    init_workspace()
    (workspace / "AGENTS.md").write_text("edited by the user")
    (workspace / "memory" / "user.md").write_text("written by the dream")
    (workspace / "memory" / "MEMORY.md").write_text("- user.md: the dream's index")
    (workspace / "skills" / "homelab" / "SKILL.md").write_text("edited skill")
    (legacy / "memory" / "user.md").write_text("old fact, changed")

    init_workspace()

    assert (workspace / "AGENTS.md").read_text() == "edited by the user"
    assert (workspace / "memory" / "user.md").read_text() == "written by the dream"
    assert (workspace / "memory" / "MEMORY.md").read_text() == "- user.md: the dream's index"
    assert (workspace / "skills" / "homelab" / "SKILL.md").read_text() == "edited skill"


def test_copied_skills_name_their_files_by_the_new_path(workspace: Path, monkeypatch) -> None:
    legacy_skills = _legacy(workspace) / "skills"
    _skill(legacy_skills, "tracker")
    (legacy_skills / "tracker" / "scripts").mkdir()
    (legacy_skills / "tracker" / "scripts" / "run.py").write_text(
        f'DATA = "{legacy_skills}/tracker/data.json"\n'
    )
    (legacy_skills / "tracker" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    init_workspace()

    script = (workspace / "skills" / "tracker" / "scripts" / "run.py").read_text()
    assert script == f'DATA = "{workspace / "skills"}/tracker/data.json"\n'
    logo = (workspace / "skills" / "tracker" / "logo.png").read_bytes()
    assert logo == b"\x89PNG\r\n\x1a\n\xff\xfe"


def test_a_copy_cut_short_is_retried_on_the_next_start(workspace: Path, monkeypatch) -> None:
    _skill(_legacy(workspace) / "skills", "homelab")
    real_copytree = shutil.copytree

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("aug.core.workspace.shutil.copytree", disk_full)
    init_workspace()  # logs and carries on
    assert not (workspace / "skills" / "homelab").exists()

    monkeypatch.setattr("aug.core.workspace.shutil.copytree", real_copytree)
    init_workspace()
    assert (workspace / "skills" / "homelab" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# session_prompt
# ---------------------------------------------------------------------------


def test_the_session_prompt_reads_the_workspace(workspace: Path) -> None:
    init_workspace()
    (workspace / "AGENTS.md").write_text("Be AUG.")
    (workspace / "memory" / "MEMORY.md").write_text("- people.md: Asya, Sasha")
    _skill(workspace / "skills", "tts", "Speak text aloud")

    prompt = session_prompt("telegram")

    assert "Be AUG." in prompt
    assert "- people.md: Asya, Sasha" in prompt
    assert "- tts: Speak text aloud" in prompt
    assert str(workspace / "memory") in prompt


def test_the_memory_index_is_capped(workspace: Path) -> None:
    init_workspace()
    lines = [f"- file{i}.md: " + "x" * 60 for i in range(100)]
    (workspace / "memory" / "MEMORY.md").write_text("\n".join(lines))

    prompt = session_prompt("telegram")

    snapshot = prompt.split("<memory_index>")[1].split("</memory_index>")[0]
    assert len(snapshot) < MEMORY_INDEX_MAX_CHARS + 200  # plus the section's indentation
    assert "file0.md" in snapshot
    assert "file99.md" not in snapshot


def test_notes_never_reach_the_prompt(workspace: Path) -> None:
    init_workspace()
    append_note("the wifi password changed")

    assert "wifi" not in session_prompt("telegram")


class _RecordingLLM:
    """Stands in for the bound model: records each call's messages, replies "ok"."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="ok")


async def test_the_prompt_is_the_same_on_every_turn_of_a_session(workspace: Path) -> None:
    init_workspace()
    agent = AugAgent(model="test-model")
    llm = _RecordingLLM()
    agent._llm = llm
    harness = LangGraphHarness(agent, MemorySaver())

    await _turn(harness, "first", "ctx-1")
    append_note("a new fact")
    (workspace / "AGENTS.md").write_text("Edited mid-chat.")
    _skill(workspace / "skills", "brand-new")
    await _turn(harness, "second", "ctx-1")
    await _turn(harness, "third", "ctx-2")  # a new session (as after /clear)

    first, second, fresh = (_system(call) for call in llm.calls)
    assert first == second
    assert "Edited mid-chat." not in second
    assert "Edited mid-chat." in fresh
    assert "brand-new" in fresh
    assert await harness.session_prompt("ctx-1", "telegram") == first


async def test_switching_to_an_agent_with_another_prompt_rebuilds_it(workspace: Path) -> None:
    init_workspace()
    checkpointer = MemorySaver()
    describing = AugAgent(model="test-model", vision_description_model="vision-model")
    seeing = AugAgent(model="test-model")
    describing._llm, seeing._llm = _RecordingLLM(), _RecordingLLM()

    await _turn(LangGraphHarness(describing, checkpointer), "first", "ctx")
    await _turn(LangGraphHarness(seeing, checkpointer), "after /version", "ctx")

    assert "describe_image" in _system(describing._llm.calls[0])
    assert "describe_image" not in _system(seeing._llm.calls[0])


async def test_session_prompt_previews_a_session_not_started_yet(workspace: Path) -> None:
    init_workspace()
    harness = LangGraphHarness(AugAgent(model="test-model"), MemorySaver())

    assert await harness.session_prompt("never-used", "telegram") == session_prompt("telegram")


# ---------------------------------------------------------------------------
# The dream writes only memory/
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_state_writes():
    with patch("aug.core.memory.save_state"):
        yield


def _dream_llm(text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=text))
    return llm


async def test_the_light_dream_writes_memory_and_its_index_only(workspace: Path) -> None:
    init_workspace()
    append_note("Vadim got a bike")
    agents_before = (workspace / "AGENTS.md").read_text()
    reply = "<user>Has a bike.</user><memory_index>- user.md: bike, London</memory_index>"

    with patch("aug.core.memory.build_chat_model", return_value=_dream_llm(reply)):
        await run_light_consolidation()

    assert (workspace / "memory" / "user.md").read_text() == "Has a bike.\n"
    assert (workspace / "memory" / "MEMORY.md").read_text() == "- user.md: bike, London\n"
    assert (workspace / "AGENTS.md").read_text() == agents_before
    outside = [p for p in workspace.rglob("*") if p.is_file() and "memory" not in p.parts]
    assert outside == [workspace / "AGENTS.md"]


async def test_the_deep_dream_writes_context_and_a_capped_index(workspace: Path) -> None:
    init_workspace()
    long_index = "\n".join(f"- f{i}.md: " + "y" * 80 for i in range(60))
    reply = (
        "<context>## Present\n[2026-11-13] PR 4</context>"
        f"<memory_index>{long_index}</memory_index><new_reflection>r</new_reflection>"
    )

    with patch("aug.core.memory.build_chat_model", return_value=_dream_llm(reply)):
        await run_deep_consolidation()

    assert "PR 4" in (workspace / "memory" / "context.md").read_text()
    index = (workspace / "memory" / "MEMORY.md").read_text()
    assert len(index) <= MEMORY_INDEX_MAX_CHARS + 1
    assert index.endswith(".md: " + "y" * 80 + "\n")  # cut at a line break


async def test_an_omitted_index_is_left_alone(workspace: Path) -> None:
    init_workspace()
    append_note("x")
    before = (workspace / "memory" / "MEMORY.md").read_text()

    with patch("aug.core.memory.build_chat_model", return_value=_dream_llm("<user>u</user>")):
        await run_light_consolidation()

    assert (workspace / "memory" / "MEMORY.md").read_text() == before


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_get_skill_gives_the_skill_folder(workspace: Path) -> None:
    _skill(workspace / "skills", "tts")

    result = get_skill.invoke({"name": "tts"})

    assert result.startswith(f"Skill folder: {(workspace / 'skills' / 'tts').resolve()}\n")
    assert "Body of tts." in result


def test_get_skill_loads_a_listed_skill_with_an_old_style_name(workspace: Path) -> None:
    _skill(workspace / "skills", "home_assistant")

    assert "Body of home_assistant." in get_skill.invoke({"name": "home_assistant"})


def test_get_skill_refuses_a_path_for_a_name(workspace: Path) -> None:
    assert get_skill.invoke({"name": "../../AGENTS"}) == "Skill '../../AGENTS' not found."


def test_rewriting_a_skill_keeps_its_other_frontmatter(workspace: Path) -> None:
    folder = workspace / "skills" / "tts"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: tts\ndescription: old\nlicense: MIT\nallowed-tools: Bash\n---\n\nold body\n"
    )

    write_skill_md(folder, "tts", "new", "new body")

    text = (folder / "SKILL.md").read_text()
    assert "license: MIT" in text
    assert "allowed-tools: Bash" in text
    detail = load_skill("tts", workspace / "skills")
    assert (detail.description, detail.body) == ("new", "new body")


async def _turn(harness: LangGraphHarness, text: str, context_id: str) -> None:
    message = Message(parts=[TextPart(text=text)], sender=Sender(interface="telegram", user_id="1"))
    events = [e async for e in harness.run(message, context_id, TURN)]
    assert isinstance(events[-1], TurnEnded) and events[-1].error is None


def _system(messages: list) -> str:
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    return messages[0].content
