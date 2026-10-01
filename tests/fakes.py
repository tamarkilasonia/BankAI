"""Test doubles."""

from collections.abc import Sequence
from typing import Any

from app.llm import ChatMessage, LLMCompletion, LLMError


class ScriptedLLM:
    """An LLMClient that replays scripted replies (or raises scripted errors) in order.

    Every call is recorded, so tests can assert what the agent asked the model.
    """

    def __init__(self, *replies: str | LLMError) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "scripted"

    async def aclose(self) -> None:
        self.closed = True

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        self.calls.append({"messages": list(messages), "json_schema": json_schema, "max_tokens": max_tokens})
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self._replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        return LLMCompletion(text=reply, model="scripted", latency_ms=1.0)
