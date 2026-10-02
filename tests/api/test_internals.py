"""The API's robustness paths: expiry, audit error codes, error mapping, middleware edge cases, shutdown."""

import asyncio
import uuid
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.agents.assistant import AssistantReply
from app.agents.intents import Intent
from app.api.conversations import ConversationStore, _error_code
from app.api.errors import install_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.models.enums import SafetyStatus
from app.services.errors import NotFoundError, ServiceUnavailableError
from tests.api.conftest import Api


def reply(**changes: Any) -> AssistantReply:
    base = AssistantReply(
        request_id="r",
        response="ok",
        intent=Intent.GENERAL,
        intent_source="rules",
        intent_confidence=1.0,
        tool=None,
        tool_arguments=None,
        tool_result=None,
        tool_error=None,
        response_source="template",
        safety_status=SafetyStatus.ALLOWED,
        safety_flags=[],
        language="en",
        llm_calls=0,
        errors=[],
        latency_ms=1.0,
        timings_ms={},
    )
    return base.model_copy(update=changes)


def test_expired_conversations_are_forgotten(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import conversations

    store = ConversationStore(timedelta(minutes=30))
    user = uuid.uuid4()
    conversation_id, _ = store.open(None, user)
    assert store.open(conversation_id, user)[0] == conversation_id

    later = conversations.utcnow() + timedelta(minutes=31)
    monkeypatch.setattr(conversations, "utcnow", lambda: later)
    new_id, conversation = store.open(conversation_id, user)

    assert new_id != conversation_id and conversation.previous is None
    assert len(store) == 1  # the expired one was purged


def test_audit_error_codes() -> None:
    assert _error_code(reply(tool_error="unavailable")) == "unavailable"
    assert _error_code(reply(errors=["llm_timeout", "internal_error"])) == "internal_error"
    assert _error_code(reply(errors=["ungrounded_answer"])) is None  # handled inside the agent: not an error


async def test_domain_errors_map_to_statuses() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/missing")
    async def missing() -> None:
        raise NotFoundError("Unknown loan product 'X'.")

    @app.get("/down")
    async def down() -> None:
        raise ServiceUnavailableError("knowledge base offline")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        not_found, unavailable = await client.get("/missing"), await client.get("/down")

    assert (not_found.status_code, not_found.json()["error"]["message"]) == (404, "Unknown loan product 'X'.")
    assert unavailable.status_code == 503
    assert "offline" not in unavailable.text  # the internal reason stays internal


async def test_middleware_passes_other_protocols_through() -> None:
    seen: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    await RequestContextMiddleware(app)({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan"]


async def test_a_crash_after_the_response_started_is_only_logged(caplog: pytest.LogCaptureFixture) -> None:
    async def streaming_then_crash(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("broke mid-stream")

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/x", "headers": []}
    await RequestContextMiddleware(streaming_then_crash)(scope, None, send)  # type: ignore[arg-type]

    assert [m["type"] for m in sent] == ["http.response.start"]  # no second, conflicting response
    assert "http.unhandled_error" in caplog.text


async def test_shutdown_cancels_a_running_evaluation(api: Api) -> None:
    async def forever() -> None:
        await asyncio.sleep(3600)

    task: asyncio.Task[object] = asyncio.create_task(forever())  # type: ignore[arg-type]
    api.services.evaluation_task = task
    await api.services.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_api_with_the_mcp_server_over_http_unreachable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, reset_app_state: None
) -> None:
    """The deployment topology (API and MCP server as separate services), with the MCP server down."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.models
    from app.api.main import create_app
    from app.database.base import Base
    from app.database.seed import seed_database
    from app.database.session import create_engine
    from tests.constants import ANCHOR

    url = f"sqlite+aiosqlite:///{tmp_path / 'bank.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        await seed_database(session, anchor=ANCHOR)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    monkeypatch.setenv("MCP_SERVER_URL", "http://127.0.0.1:9/mcp")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "2")

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        login = await client.post("/auth/login", json={"customer_number": "CUST-0001"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        body = (await client.post("/chat", json={"message": "What's my balance?"}, headers=headers)).json()
        health = (await client.get("/health")).json()

    assert body["trace"]["tool_error"] == "unavailable"
    assert body["response"] == "Your banking data is temporarily unavailable. Please try again in a moment."
    assert health["components"]["tools"] == "mcp (http)"
