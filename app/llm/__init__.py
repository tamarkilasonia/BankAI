"""Language-model access, behind a small interface so providers are swappable."""

from app.core.config import Settings
from app.llm.base import (
    ChatMessage,
    LLMClient,
    LLMCompletion,
    LLMError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    complete_structured,
)


def create_llm_client(settings: Settings) -> LLMClient | None:
    """The configured LLM, or None for LLM_PROVIDER=none (rules + templates only)."""
    if settings.llm_provider == "none":
        return None
    from app.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient.from_settings(settings)


__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMCompletion",
    "LLMError",
    "LLMOutputError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "complete_structured",
    "create_llm_client",
]
