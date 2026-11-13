"""Shared LangGraph state definition used by all agents."""

from typing import Annotated

from langchain_core.messages import AnyMessage, RemoveMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(BaseModel):
    """State that flows through every node in the graph.

    messages:      Full conversation history, managed by LangGraph's
                   add_messages reducer (appends rather than replaces).
    thread_id:     Opaque identifier propagated from the API layer.
    session_prompt: The system prompt, built by preprocess() on a session's first turn
                   and kept for the rest of it, so the cached prefix never changes
                   mid-chat.  None until then.  It must default to None: LangGraph
                   writes every non-None field a node returns, so an "" default would
                   wipe it on the next node.
    session_prompt_key: Which agent setup built session_prompt (BaseAgent.prompt_key).
                   A chat switched to an agent whose prompt differs (/version) gets
                   it rebuilt once, on its next turn.
    interface:     The frontend surface ("telegram", "rest_api", "webhook", …).
                   Used to look up interface-specific prompts at runtime.
    """

    messages: Annotated[list[AnyMessage], add_messages] = []
    thread_id: str = ""
    session_prompt: str | None = None
    session_prompt_key: str | None = None
    interface: str = ""

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class AgentStateUpdate(AgentState):
    """Partial state returned by a node.

    Identical to AgentState but semantically signals that only the fields
    explicitly set will be merged into the graph state.

    messages may include RemoveMessage entries — these are reducer instructions
    consumed by add_messages and never stored in state.
    """

    messages: Annotated[list[AnyMessage | RemoveMessage], add_messages] = []
