"""LLM factory.

All LLM access in AUG goes through this module.  Every call uses ChatOpenAI
pointed at the LiteLLM proxy — AUG itself has zero provider-switching logic.
LiteLLM handles routing to the real provider (OpenAI, Anthropic, etc.).

Which model to use is defined *per agent* via that agent's AGENT_CONFIG dict,
not via env vars.  This lets different agents use different models without any
application-level changes.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from aug.config import settings


def build_llm(config: dict) -> BaseChatModel:
    """Build and return a chat model from *config*.

    Expected keys in *config*:
        model (str): any model string that LiteLLM understands, e.g. "gpt-4o-mini".
        temperature (float, optional): defaults to 0.7.

    The client always targets the LiteLLM proxy defined in settings.
    """
    return ChatOpenAI(
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
        base_url=settings.LLM_BASE_URL,
    )
