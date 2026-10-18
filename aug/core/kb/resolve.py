"""Resolution: names to entities, predicate names to the registry, and the
searchable profile each entity carries.

The cascade stops as early as it can — exact name, alias, name-token subset, then
trigram — and a genuine tie is never guessed: it comes back as ambiguous.
"""

import re
from dataclasses import dataclass
from typing import Literal

from aug.utils.kb.embeddings import EmbeddingClient
from aug.utils.kb.store import EntityRow, KbSession, is_object_id

ME_ID = "person:me"
ME_NAME = "Me (the user)"
# ≈4k tokens — well inside the embedding model's input limit.
MAX_PROFILE_CHARS = 16_000

_TRIGRAM_THRESHOLD = 0.55

# Words that name the shape of a value rather than what it is about: "phone" and
# "phone_number" are one predicate.  Anything else that differs — "net" vs "gross",
# "home" vs "work", "previous" — makes a different predicate.  Meaning-level
# matching is deliberately absent: a wrong merge into a single-valued predicate
# silently overwrites a value, a near-duplicate predicate costs nothing.
_PREDICATE_FILLER = frozenset({"number", "no", "num", "id", "name", "value", "code"})


@dataclass(frozen=True)
class EntityResolution:
    entity_id: str | None  # None when ambiguous
    created: bool = False
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredicateResolution:
    name: str
    status: Literal["matched", "registered"]
    single_valued: bool


def normalize_predicate(name: str) -> str:
    name = re.sub(r"[^\w]+", "_", name.strip().lower()).strip("_")
    return re.sub(r"_+", "_", name) or "related_to"


def humanize_predicate(name: str) -> str:
    return name.replace("_", " ")


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-")
    return slug or "unnamed"


async def ensure_me(kb: KbSession) -> None:
    if not await kb.entity_id_taken(ME_ID):
        await kb.insert_entity(ME_ID, "person", ME_NAME, [])


async def resolve_entity(
    kb: KbSession,
    *,
    name: str,
    entity_type: str,
    aliases: list[str],
    existing_id: str | None,
    candidates: list[str],
) -> EntityResolution:
    """Find the entity *name* refers to, creating it when it is genuinely new."""
    if existing_id and existing_id != ME_ID and await kb.get_entity(existing_id):
        await _learn_names(kb, existing_id, name, aliases)
        return EntityResolution(existing_id)
    if existing_id == ME_ID:
        await ensure_me(kb)
        return EntityResolution(ME_ID)

    live_candidates = [c for c in candidates if await kb.get_entity(c)]
    if len(live_candidates) >= 2:
        return EntityResolution(None, candidates=tuple(live_candidates))

    matches = await _cascade(kb, name, entity_type)
    if len(matches) == 1:
        await _learn_names(kb, matches[0].id, name, aliases)
        return EntityResolution(matches[0].id)
    if len(matches) > 1:
        return EntityResolution(None, candidates=tuple(m.id for m in matches))

    entity_id = f"{entity_type}:{slugify(name)}"
    suffix = 2
    while await kb.entity_id_taken(entity_id):
        entity_id = f"{entity_type}:{slugify(name)}-{suffix}"
        suffix += 1
    await kb.insert_entity(entity_id, entity_type, name, _clean_aliases(name, aliases))
    return EntityResolution(entity_id, created=True)


async def resolve_predicate(
    kb: KbSession,
    raw_name: str,
    *,
    value_type: str,
    single_valued: bool,
    description: str = "",
) -> PredicateResolution:
    """Reuse the registry's predicate for *raw_name*, or register it.

    Matches the exact name, else a registered name made of the same words in any
    order, ignoring filler like "number" (``number_phone`` → ``phone``).
    """
    name = normalize_predicate(raw_name)
    existing = await kb.get_predicate(name)
    if existing is None:
        key = predicate_key(name)
        existing = next(
            (p for p in await kb.list_predicates() if predicate_key(p.name) == key), None
        )
    if existing:
        await kb.bump_predicate(existing.name)
        return PredicateResolution(existing.name, "matched", existing.single_valued)
    await kb.insert_predicate(
        name, description=description, value_type=value_type, single_valued=single_valued
    )
    await kb.bump_predicate(name)
    return PredicateResolution(name, "registered", single_valued)


def predicate_key(name: str) -> str:
    """Order- and filler-insensitive identity of a normalised predicate name."""
    words = [_singular(w) for w in name.split("_") if w]
    meaningful = sorted({w for w in words if w not in _PREDICATE_FILLER})
    return "_".join(meaningful or sorted(set(words)))


async def refresh_profiles(kb: KbSession, embedder: EmbeddingClient, entity_ids: set[str]) -> None:
    """Rebuild the searchable profile row of each entity from its current facts."""
    profiles = await build_profiles(kb, entity_ids)
    if not profiles:
        return
    vectors = await embedder.embed([text for _, text in profiles])
    for (entity_id, text), vector in zip(profiles, vectors, strict=True):
        await kb.replace_profile(entity_id, text, vector)


async def build_profiles(kb: KbSession, entity_ids: set[str]) -> list[tuple[str, str]]:
    """(entity id, profile text) for each live entity among ``entity_ids``, unembedded —
    for a caller that embeds them alongside other text in one call."""
    profiles = []
    for entity_id in sorted(e for e in entity_ids if e and not is_object_id(e)):
        entity = await kb.get_entity(entity_id)
        if entity is not None:
            profiles.append((entity_id, await build_profile(kb, entity)))
    return profiles


async def build_profile(kb: KbSession, entity: EntityRow) -> str:
    """Name, aliases and current facts, one per line — capped, since an entity like "me"
    gathers facts without end and the whole profile goes to the embedder."""
    head = f"{entity.display_name} ({entity.type}, id {entity.id})"
    if entity.aliases:
        head += f" — also called {', '.join(entity.aliases)}"
    lines = [head]
    for fact in await kb.facts_out(entity.id, None):
        value = fact.value_literal
        if fact.value_node_id:
            value = f"{await kb.node_label(fact.value_node_id)} [{fact.value_node_id}]"
        lines.append(f"{humanize_predicate(fact.predicate)}: {value}")
    for fact in await kb.facts_in(entity.id, None):
        if is_object_id(fact.subject_id):
            continue
        subject = await kb.node_label(fact.subject_id)
        lines.append(f"{subject} [{fact.subject_id}] — {humanize_predicate(fact.predicate)}")
    return "\n".join(lines)[:MAX_PROFILE_CHARS]


async def _cascade(kb: KbSession, name: str, entity_type: str) -> list[EntityRow]:
    exact = [e for e in await kb.entities_named(name) if e.id != ME_ID]
    if exact:
        return exact
    tokens = _name_tokens(name)
    if tokens:
        subset = [
            e
            for e in await kb.list_entities()
            if e.id != ME_ID
            and e.type == entity_type
            and (tokens <= _name_tokens(e.display_name) or _name_tokens(e.display_name) <= tokens)
        ]
        if subset:
            return subset
    return [
        e for e in await kb.entities_similar(name, entity_type, _TRIGRAM_THRESHOLD) if e.id != ME_ID
    ]


async def _learn_names(kb: KbSession, entity_id: str, name: str, aliases: list[str]) -> None:
    entity = await kb.get_entity(entity_id)
    if entity is None:
        return
    known = {entity.display_name.lower(), *(a.lower() for a in entity.aliases)}
    new = [a for a in _clean_aliases("", [name, *aliases]) if a.lower() not in known]
    if new:
        await kb.add_aliases(entity_id, new)


def _clean_aliases(name: str, aliases: list[str]) -> list[str]:
    seen = {name.lower()} if name else set()
    out = []
    for alias in aliases:
        alias = alias.strip()
        if alias and alias.lower() not in seen:
            seen.add(alias.lower())
            out.append(alias)
    return out


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"\w+", name.lower(), flags=re.UNICODE) if len(t) > 1}


def _singular(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word
