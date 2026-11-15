"""DecisionsClient: the request it sends and how it reads the reply."""

import json
from unittest.mock import patch

import httpx
import pytest

from aug.utils.decisions import (
    Answer,
    Choice,
    DecisionsClient,
    DecisionsError,
    InputMessage,
    Predicate,
)

_REAL_CLIENT = httpx.AsyncClient


@pytest.fixture
def api():
    """Serves the client's requests from ``api.handler``; ``api.seen`` lists them."""

    class _Api:
        handler = staticmethod(lambda request: httpx.Response(500))
        seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        _Api.seen.append(request)
        return _Api.handler(request)

    transport = httpx.MockTransport(record)
    with patch(
        "aug.utils.decisions.httpx.AsyncClient",
        side_effect=lambda **kw: _REAL_CLIENT(transport=transport, **kw),
    ):
        yield _Api


def _client() -> DecisionsClient:
    return DecisionsClient("http://litellm:4000/v1", "sk-aug", model="gpt-6-luna")


async def test_questions_go_to_the_proxys_pass_through_and_answers_come_back_by_name(api):
    reply = {
        "answers": [
            {"type": "predicate", "name": "gate", "probability": 0.97},
            {"type": "choice", "name": "action", "choice": "turn_off", "confidence": 0.9,
             "probabilities": []},
            {"type": "refusal", "name": "colour"},
        ]
    }  # fmt: skip
    api.handler = lambda r: httpx.Response(200, json=reply)

    answers = await _client().decide(
        [InputMessage("assistant", "Kitchen is on."), InputMessage("user", "turn it off")],
        [
            Predicate("gate", "A request?"),
            Choice("action", "Which?", {"turn_on": "On.", "turn_off": "Off."}),
            Choice("colour", "Colour?", {"red": "red"}),
        ],
    )

    assert answers == {
        "gate": Answer(probability=0.97),
        "action": Answer(choice="turn_off", confidence=0.9),
        "colour": Answer(),
    }
    (request,) = api.seen
    assert str(request.url) == "http://litellm:4000/openai_passthrough/v1/decisions"
    assert request.headers["Authorization"] == "Bearer sk-aug"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": [
            {"role": "assistant", "content": "Kitchen is on."},
            {"role": "user", "content": "turn it off"},
        ],
        "questions": [
            {"type": "predicate", "name": "gate", "instructions": "A request?"},
            {
                "type": "choice",
                "name": "action",
                "instructions": "Which?",
                "choices": [
                    {"value": "turn_on", "description": "On."},
                    {"value": "turn_off", "description": "Off."},
                ],
            },
            {
                "type": "choice",
                "name": "colour",
                "instructions": "Colour?",
                "choices": [{"value": "red", "description": "red"}],
            },
        ],
    }


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="Decision admission timed out."),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"no": "answers"}),
    ],
)
async def test_a_failed_or_unreadable_reply_raises(api, response):
    api.handler = lambda r: response
    with pytest.raises(DecisionsError):
        await _client().decide([InputMessage("user", "hi")], [Predicate("g", "?")])


async def test_an_unreachable_api_raises(api):
    def down(request):
        raise httpx.ConnectError("refused")

    api.handler = down
    with pytest.raises(DecisionsError):
        await _client().decide([InputMessage("user", "hi")], [Predicate("g", "?")])
