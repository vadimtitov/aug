"""OpenAI's Decisions API, through the LiteLLM proxy's OpenAI pass-through.

A decisions model answers typed questions about a conversation with probabilities
instead of text: a ``Predicate`` gets the chance it is true, a ``Choice`` gets the most
likely of its options.  It is not a chat model, so it is reached here rather than
through ``build_chat_model()``; the proxy still holds the OpenAI key and does the
accounting.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

_PATH = "/openai_passthrough/v1/decisions"


@dataclass(frozen=True)
class InputMessage:
    role: Literal["developer", "user", "assistant"]
    text: str


@dataclass(frozen=True)
class Predicate:
    """A yes/no question: answered with the probability it is true."""

    name: str
    instructions: str


@dataclass(frozen=True)
class Choice:
    """One of *options* (value -> description): answered with the most likely value."""

    name: str
    instructions: str
    options: Mapping[str, str]


Question = Predicate | Choice


@dataclass(frozen=True)
class Answer:
    """probability: a Predicate's.  choice/confidence: a Choice's.  Both None when the
    model refused the question."""

    probability: float | None = None
    choice: str | None = None
    confidence: float | None = None


class DecisionsError(Exception):
    """The API could not be reached or returned an error."""


class DecisionsClient:
    """Asks a decisions model questions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 4.0) -> None:
        """*base_url*: the proxy's, with or without its ``/v1`` suffix."""
        self._url = base_url.rstrip("/").removesuffix("/v1") + _PATH
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._timeout = timeout

    async def decide(
        self, conversation: Sequence[InputMessage], questions: Sequence[Question]
    ) -> dict[str, Answer]:
        """Answer *questions* about *conversation*, keyed by question name.

        Raises:
            DecisionsError: the request failed or the reply could not be read.
        """
        body = {
            "model": self._model,
            "input": [{"role": m.role, "content": m.text} for m in conversation],
            "questions": [_question(q) for q in questions],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                response = await http.post(self._url, headers=self._headers, json=body)
        except httpx.HTTPError as e:
            raise DecisionsError(f"request failed: {e!r}") from e
        if response.is_error:
            raise DecisionsError(f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            return {a["name"]: _answer(a) for a in response.json()["answers"]}
        except (ValueError, KeyError, TypeError) as e:
            raise DecisionsError(f"unreadable reply: {response.text[:300]}") from e


def _question(q: Question) -> dict:
    if isinstance(q, Predicate):
        return {"type": "predicate", "name": q.name, "instructions": q.instructions}
    return {
        "type": "choice",
        "name": q.name,
        "instructions": q.instructions,
        "choices": [{"value": v, "description": d} for v, d in q.options.items()],
    }


def _answer(a: dict) -> Answer:
    match a.get("type"):
        case "predicate":
            return Answer(probability=float(a["probability"]))
        case "choice":
            return Answer(choice=str(a["choice"]), confidence=float(a["confidence"]))
        case _:  # "refusal"
            return Answer()
