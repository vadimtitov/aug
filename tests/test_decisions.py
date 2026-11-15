"""DecisionsClient: the request it sends and how it reads the reply."""

import json

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


def _client(handler) -> tuple[DecisionsClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = DecisionsClient("http://litellm:4000/v1", "sk-aug", model="gpt-6-luna")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return client, seen


async def test_questions_go_to_the_proxys_pass_through_and_answers_come_back_by_name():
    reply = {
        "answers": [
            {"type": "predicate", "name": "gate", "probability": 0.97},
            {"type": "choice", "name": "action", "choice": "turn_off", "confidence": 0.9,
             "probabilities": []},
            {"type": "refusal", "name": "colour"},
        ]
    }  # fmt: skip
    client, seen = _client(lambda r: httpx.Response(200, json=reply))

    answers = await client.decide(
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
    assert answers["colour"].refused
    (request,) = seen
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
async def test_a_failed_or_unreadable_reply_raises(response):
    client, _ = _client(lambda r: response)
    with pytest.raises(DecisionsError):
        await client.decide([InputMessage("user", "hi")], [Predicate("g", "?")])


async def test_an_unreachable_api_raises():
    def down(request):
        raise httpx.ConnectError("refused")

    client, _ = _client(down)
    with pytest.raises(DecisionsError):
        await client.decide([InputMessage("user", "hi")], [Predicate("g", "?")])
