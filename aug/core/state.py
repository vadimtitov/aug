"""Shared LangGraph state definition used by all agents."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(BaseModel):
    """State that flows through every node in the graph.

    messages:      Full conversation history, managed by LangGraph's
                   add_messages reducer (appends rather than replaces).
    thread_id:     Opaque identifier propagated from the API layer.
    system_prompt: Built dynamically by preprocess(); read by respond().
                   Empty by default — respond() skips it if not set.
    """

    messages: Annotated[list[AnyMessage], add_messages] = []
    thread_id: str = ""
    system_prompt: str = ""
    interface_context: str = ""  # injected by the frontend; appended to system prompt

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class AgentStateUpdate(AgentState):
    """Partial state returned by a node.

    Identical to AgentState but semantically signals that only the fields
    explicitly set will be merged into the graph state.
    """
