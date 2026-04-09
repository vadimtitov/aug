"""Context compaction for long-running conversations.

When a thread's message history approaches the context window limit, this module
replaces old messages with an LLM-generated narrative summary.

Algorithm
---------
1. Split history into "pre-run" (everything before the last HumanMessage) and
   "current run" (last HumanMessage + everything after it — the agent's active work).
2. Decide scope:
   - If current run is small (< 50% of context window): summarise pre-run only,
     keep current run verbatim.
   - If current run is already heavy (>= 50%): summarise everything except the
     last HumanMessage itself (we keep the question so the agent knows what it
     was working on).
3. One LLM call with max_tokens=max_summary_tokens produces a dense summary.
4. Summarised messages are removed from state via RemoveMessage; summary is
   prepended as a SystemMessage.
"""

import logging
import uuid

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage

from aug.core.events import dispatch_status
from aug.core.llm import build_chat_model
from aug.core.prompts import COMPACTION_LOOP_GUARD, COMPACTION_PROMPT

logger = logging.getLogger(__name__)


def count_tokens(messages: list[AnyMessage]) -> int:
    """Estimate token count using a chars/4 heuristic.

    Accurate enough for threshold detection — we don't need precision, just
    a cheap signal that the context window is getting full.
    """
    return sum(len(str(m.content)) for m in messages) // 4


async def compact_messages(
    messages: list[AnyMessage],
    compaction_model: str,
    context_window: int,
    max_summary_tokens: int = 2000,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Compact message history by summarising old content.

    Args:
        messages: Full message list (after orphan cleanup).
        compaction_model: LiteLLM model name to use for summarisation.
        context_window: Token limit of the main model (used for the 50% decision).
        max_summary_tokens: Hard cap on summary length, passed as max_tokens to
            the summarisation LLM call.

    Returns:
        (messages_for_llm, state_changes)
        - messages_for_llm: compacted view to pass to the main LLM
        - state_changes: RemoveMessage entries + SystemMessage(summary) to
          return in AgentStateUpdate so the checkpointer persists the cleanup.
          Empty list if nothing was compacted.
    """
    last_human_idx = _last_human_index(messages)
    if last_human_idx is None:
        return messages, []

    pre_run = messages[:last_human_idx]
    current_run = messages[last_human_idx:]  # includes the last HumanMessage

    tokens_current = count_tokens(current_run)
    # "Heavy" if current run exceeds 50% of context, OR if there's no pre-run at all.
    # In the no-pre-run case the current run IS everything; the caller already verified
    # the total token count exceeds the compaction threshold.
    current_run_heavy = tokens_current >= context_window * 0.5 or not pre_run

    if current_run_heavy:
        # Summarise pre-run + current run tool work; keep only the last HumanMessage
        to_summarise = pre_run + current_run[1:]  # everything except last HumanMessage
        to_keep = [current_run[0]]  # just the last HumanMessage
        scope = "full (current run too heavy)"
    else:
        # Summarise pre-run only; keep current run intact
        to_summarise = pre_run
        to_keep = current_run
        scope = "pre-run only"

    if not to_summarise:
        return messages, []

    logger.info(
        "compaction triggered scope=%s summarising=%d messages keeping=%d messages",
        scope,
        len(to_summarise),
        len(to_keep),
    )

    await dispatch_status("🗜 Compacting conversation…")
    summary = await _summarise(to_summarise, compaction_model, max_summary_tokens)

    logger.info(
        "compaction complete summary_chars=%d removed=%d",
        len(summary),
        len([m for m in to_summarise if m.id]),
    )

    summary_message = SystemMessage(content=summary, id=str(uuid.uuid4()))

    state_changes: list[AnyMessage] = [RemoveMessage(id=m.id) for m in to_summarise if m.id] + [
        summary_message
    ]

    # If a previous summary is being re-compacted, we are in a research loop.
    # Inject a one-off synthesis instruction (not persisted) so the LLM stops
    # calling tools and produces its answer instead.
    already_compacted = any(isinstance(m, SystemMessage) for m in to_summarise)
    if already_compacted:
        logger.warning("compaction loop detected — injecting synthesis guard")
        loop_guard = SystemMessage(content=COMPACTION_LOOP_GUARD, id=str(uuid.uuid4()))
        messages_for_llm = [summary_message, loop_guard, *to_keep]
    else:
        messages_for_llm = [summary_message, *to_keep]

    return messages_for_llm, state_changes


async def _summarise(messages: list[AnyMessage], model: str, max_tokens: int) -> str:
    llm = build_chat_model(model, temperature=0.2, max_tokens=max_tokens)
    history = _format_for_summary(messages)
    # Pass empty callbacks to prevent this internal LLM call from emitting
    # on_chat_model_stream events into the parent graph's event stream.
    response = await llm.ainvoke(
        COMPACTION_PROMPT.format(history=history),
        config={"callbacks": []},
    )
    return f"[Conversation summary]:\n{response.content}"


def _last_human_index(messages: list[AnyMessage]) -> int | None:
    indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    return indices[-1] if indices else None


def _format_for_summary(messages: list[AnyMessage]) -> str:
    parts = []
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)
