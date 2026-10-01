"""LLM client for any server that speaks the OpenAI Chat Completions API.

That includes OpenAI itself and local servers such as Ollama, which exposes the
same API at ``http://localhost:11434/v1``. Switching between them is
configuration only (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY).
"""

import time
from collections.abc import Sequence
from typing import Any

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import ConfigurationError, Settings
from app.llm.base import ChatMessage, LLMCompletion, LLMTimeoutError, LLMUnavailableError

# Ollama ignores the key but the OpenAI SDK insists on having one.
_PLACEHOLDER_KEY = "not-needed-for-local-servers"


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if base_url is None and not api_key:
            raise ConfigurationError(
                "LLM_API_KEY is required when calling api.openai.com (LLM_BASE_URL unset)."
            )
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or _PLACEHOLDER_KEY,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=1,  # one retry for transient errors; the agent has its own fallbacks
            http_client=http_client,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleClient":
        return cls(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": max_tokens or self._max_output_tokens,
        }
        if json_schema is not None:
            # Structured outputs: the server constrains generation to this schema
            # (supported by OpenAI and Ollama). We still validate the result.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema},
            }

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request)
        except APITimeoutError as exc:
            raise LLMTimeoutError("the language model did not answer in time") from exc
        except APIConnectionError as exc:
            raise LLMUnavailableError("the language model server is unreachable") from exc
        except APIStatusError as exc:
            raise LLMUnavailableError(f"the language model server returned HTTP {exc.status_code}") from exc

        usage = response.usage
        text = (response.choices[0].message.content or "") if response.choices else ""
        return LLMCompletion(
            text=text,
            model=response.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
