"""ScriptedHarness: a Harness that plays back fixed events, for tests above the harness."""

from collections.abc import AsyncIterator

from aug.core.harness import Harness, HarnessEvent, Message, ModelTier, TurnContext


class ScriptedHarness(Harness):
    """Yields *events* for every turn and records what callers asked of it.

    ``accept_inject`` decides what inject() answers, as if a turn were (or weren't)
    in flight.
    """

    def __init__(
        self,
        events: list[HarnessEvent] | None = None,
        accept_inject: bool = False,
        running: bool = True,
    ):
        self.events = list(events or [])
        self.accept_inject = accept_inject
        self.running_turn = running  # what cancel() answers
        self.turns: list[tuple[Message, str, TurnContext]] = []
        self.injected: list[tuple[Message, str]] = []
        self.cancelled: list[str] = []
        self.compacted: list[str] = []

    async def _run(
        self, message: Message, context_id: str, turn: TurnContext
    ) -> AsyncIterator[HarnessEvent]:
        self.turns.append((message, context_id, turn))
        for event in self.events:
            yield event

    async def inject(self, message: Message, context_id: str) -> bool:
        if self.accept_inject:
            self.injected.append((message, context_id))
        return self.accept_inject

    async def cancel(self, context_id: str) -> bool:
        self.cancelled.append(context_id)
        return self.running_turn

    async def compact(self, context_id: str) -> bool:
        self.compacted.append(context_id)
        return True

    async def session_prompt(self, context_id: str, interface: str) -> str:
        return f"prompt:{context_id}:{interface}"

    async def oneshot(self, prompt: str, tier: ModelTier = ModelTier.STRONG) -> str:
        return f"oneshot:{prompt}"
