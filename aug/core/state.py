"""Shared LangGraph state definition used by all agents."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that flows through every node in the graph.

    messages: the full conversation history, managed by LangGraph's
              add_messages reducer (appends rather than replaces).
    thread_id: opaque identifier propagated from the API layer so nodes
               can use it for memory lookups, logging, etc.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str


class AgentStateUpdate(TypedDict, total=False):
    """Partial state returned by a node — only the fields being updated.

    ``total=False`` makes all keys optional so nodes can return just
    ``{"messages": [...]}`` without having to supply ``thread_id``.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
