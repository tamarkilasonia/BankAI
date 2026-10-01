"""The LLM interface the agent depends on.

The agent needs exactly two things from a language model:

* ``complete``            - free text (phrasing an answer from facts), and
* ``complete_structured`` - JSON that must validate against a Pydantic model
                            (classifying an intent and extracting tool arguments).

Anything implementing ``LLMClient`` can be plugged in: the OpenAI-compatible
client (OpenAI or a local Ollama) in production, scripted fakes in tests.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMError(Exception):
    """The LLM call did not produce a usable result. The agent falls back, it never crashes."""

    code = "llm_error"


class LLMUnavailableError(LLMError):
    code = "llm_unavailable"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"


class LLMOutputError(LLMError):
    """The model answered, but not with valid JSON for the requested schema."""

    code = "llm_invalid_output"


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        """Return the model's reply. With ``json_schema`` the reply must be JSON for that schema."""
        ...

    async def aclose(self) -> None:
        """Release network connections. Call once when the client is no longer needed."""
        ...


async def complete_structured[T: BaseModel](
    llm: LLMClient, messages: Sequence[ChatMessage], schema: type[T], *, max_tokens: int | None = None
) -> tuple[T, LLMCompletion]:
    """Ask for JSON matching ``schema`` and validate it - never trust the raw model output."""
    completion = await llm.complete(messages, json_schema=schema.model_json_schema(), max_tokens=max_tokens)
    try:
        return schema.model_validate_json(_strip_code_fence(completion.text)), completion
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMOutputError(f"model output did not match {schema.__name__}") from exc


def _strip_code_fence(text: str) -> str:
    """Small models sometimes wrap JSON in ```json fences despite being asked not to."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()
