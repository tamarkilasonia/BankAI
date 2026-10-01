"""The OpenAI-compatible client, tested against a mocked HTTP transport (no network, no key)."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel

from app.core.config import ConfigurationError, Settings
from app.llm import (
    ChatMessage,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    complete_structured,
    create_llm_client,
)
from app.llm.openai_compatible import OpenAICompatibleClient
from tests.fakes import ScriptedLLM

MESSAGES = [ChatMessage("system", "be brief"), ChatMessage("user", "hi")]


def chat_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen2.5:3b",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def client_with(handler: Callable[[httpx2.Request], httpx2.Response]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model="qwen2.5:3b",
        base_url="http://ollama.test/v1",
        timeout_seconds=1,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def test_sends_an_openai_style_request_and_parses_the_reply() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=chat_response("Hello!"))

    completion = await client_with(handler).complete(MESSAGES, max_tokens=50)

    assert seen["url"] == "http://ollama.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:3b"
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] == 50
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("Hello!", 12, 3)


async def test_json_schema_is_requested_as_structured_output() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=chat_response("{}"))

    await client_with(handler).complete(MESSAGES, json_schema={"title": "Routing", "type": "object"})

    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "Routing"


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda request: httpx2.Response(500, json={"error": {"message": "boom"}}), LLMUnavailableError),
        (
            lambda request: httpx2.Response(404, json={"error": {"message": "model not found"}}),
            LLMUnavailableError,
        ),
    ],
)
async def test_http_errors_become_llm_errors(
    handler: Callable[[httpx2.Request], httpx2.Response], error: type[Exception]
) -> None:
    with pytest.raises(error):
        await client_with(handler).complete(MESSAGES)


async def test_connection_failure_is_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LLMUnavailableError, match="unreachable"):
        await client_with(handler).complete(MESSAGES)


async def test_timeout_is_reported_as_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMTimeoutError):
        await client_with(handler).complete(MESSAGES)


def test_openai_without_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        OpenAICompatibleClient(model="gpt-4.1-mini", base_url=None, api_key=None)


def test_factory_returns_none_offline_and_a_client_otherwise() -> None:
    assert create_llm_client(Settings(_env_file=None, llm_provider="none")) is None

    client = create_llm_client(
        Settings(_env_file=None, llm_provider="openai_compatible", llm_base_url="http://localhost:11434/v1")
    )
    assert client is not None and client.model == "qwen2.5:3b"


# --- complete_structured -------------------------------------------------------------------


class Answer(BaseModel):
    intent: str
    confidence: float


async def test_structured_output_is_validated() -> None:
    llm = ScriptedLLM('{"intent": "balance", "confidence": 0.9}')

    parsed, _ = await complete_structured(llm, MESSAGES, Answer)

    assert parsed == Answer(intent="balance", confidence=0.9)
    assert llm.calls[0]["json_schema"]["title"] == "Answer"


async def test_code_fenced_json_is_accepted() -> None:
    llm = ScriptedLLM('```json\n{"intent": "x", "confidence": 1}\n```')
    parsed, _ = await complete_structured(llm, MESSAGES, Answer)
    assert parsed.intent == "x"


@pytest.mark.parametrize(
    "reply", ["not json", '{"intent": "x"}', '{"intent": "x", "confidence": "high"}', "```"]
)
async def test_invalid_structured_output_is_an_llm_output_error(reply: str) -> None:
    with pytest.raises(LLMOutputError):
        await complete_structured(ScriptedLLM(reply), MESSAGES, Answer)
