"""KbAugAgent — AugAgent with the knowledge base.

Differs from AugAgent in two ways, both per turn:
- the KB tools are rebound every model call, so kb_query's predicate enum always
  matches the owner's current vocabulary;
- the system prompt gains the KB instructions and a compact index of the entities
  already known, so the model knows when there is something to look up.

A turn that has already made _KB_CALLS_PER_TURN KB calls loses the KB tools for the
rest of the turn, so a model stuck re-checking and re-writing must stop and answer.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from aug.core.agents.chat_agent import AugAgent
from aug.core.kb.service import get_knowledge_base
from aug.core.prompts import KB_INDEX_HEADER, KB_SYSTEM_PROMPT
from aug.core.state import AgentState, AgentStateUpdate
from aug.core.tools.kb import KB_TOOLS, kb_tools_for_turn

logger = logging.getLogger(__name__)

_INDEX_LIMIT = 150
_KB_CALLS_PER_TURN = 15


class KbAugAgent(AugAgent):
    def __init__(self, model: str, *, tools: list[BaseTool] | None = None, **kwargs) -> None:
        names = {t.name for t in KB_TOOLS}
        base = [t for t in tools or [] if t.name not in names]
        super().__init__(model, tools=[*base, *KB_TOOLS], **kwargs)

    async def preprocess(self, state: AgentState, config: RunnableConfig) -> AgentStateUpdate:
        update = super().preprocess(state)
        sections = [update.system_prompt, f"<knowledge_base>\n{KB_SYSTEM_PROMPT}\n"]
        index = await self._entity_index(config)
        if index:
            sections[-1] += f"\n{index}\n"
        sections[-1] += "</knowledge_base>"
        return update.model_copy(update={"system_prompt": "\n\n".join(sections)})

    async def respond(self, state: AgentState, config: RunnableConfig) -> AgentStateUpdate:
        owner = _owner(config)
        kb_names = {t.name for t in KB_TOOLS}
        used = _kb_calls_this_turn(state, kb_names)
        if used >= _KB_CALLS_PER_TURN:
            logger.warning("kb_call_cap_reached owner=%s calls=%d", owner, used)
            tools = [t for t in self.tools if t.name not in kb_names]
        else:
            kb_tools = await kb_tools_for_turn(get_knowledge_base(), owner)
            by_name = {t.name: t for t in kb_tools}
            tools = [by_name.get(t.name, t) for t in self.tools]
        llm = self._model.bind_tools(tools) if tools else self._model
        return await self._respond_with(llm, state)

    async def _entity_index(self, config: RunnableConfig) -> str:
        kb, owner = get_knowledge_base(), _owner(config)
        if kb is None or not owner:
            return ""
        try:
            entities = await kb.entities(owner, _INDEX_LIMIT)
        except Exception:
            logger.warning("kb_index_unavailable owner=%s", owner, exc_info=True)
            return ""
        if not entities:
            return ""
        lines = [f"- {e.display_name} [{e.id}]" for e in entities]
        return "\n".join([KB_INDEX_HEADER.format(count=len(entities)), *lines])


def _owner(config: RunnableConfig) -> str:
    configurable = config.get("configurable") or {}
    return configurable.get("user_id") or configurable.get("sender_id") or ""


def _kb_calls_this_turn(state: AgentState, kb_names: set[str]) -> int:
    count = 0
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            count += sum(1 for call in message.tool_calls if call["name"] in kb_names)
    return count
