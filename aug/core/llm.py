"""LLM factory.

All LLM access in AUG goes through this module.  Every call uses ChatOpenAI
pointed at the LiteLLM proxy — AUG itself has zero provider-switching logic.
LiteLLM handles routing to the real provider (OpenAI, Anthropic, etc.).
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from aug.config import settings


def build_chat_model(
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    top_p: float | None = None,
) -> BaseChatModel:
    """Build and return a chat model.

    Args:
        model: Any model string LiteLLM understands, e.g. ``"gpt-4o"``.
        temperature: Sampling temperature. Higher = more creative.
        max_tokens: Maximum tokens in the response.
        max_retries: Number of automatic retries on transient errors.
        timeout: Request timeout in seconds.
        seed: Fixed seed for deterministic outputs (where supported).
        stop: List of stop sequences.
        frequency_penalty: Penalise repeated tokens (-2.0 to 2.0).
        presence_penalty: Penalise tokens already in context (-2.0 to 2.0).
        top_p: Nucleus sampling probability mass.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        seed=seed,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
        base_url=settings.LLM_BASE_URL,
    )
