"""The MCP server over real streamable HTTP, plus transport failure modes."""

import asyncio
import json
import runpy
import socket
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import httpx
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from app.agents.assistant import BankingAssistant
from app.agents.nodes import AgentDeps
from app.core.config import Settings
from app.mcp import server as server_module
from app.mcp.client import MCPToolGateway, ToolCallError
from app.mcp.server import create_http_app
from app.services.authorization import Principal
from app.services.schemas import BalanceResult
from tests.constants import ANCHOR, GIORGI_CHECKING, NINO, TEST_MCP_SECRET
from tests.fakes import ScriptedLLM
from tests.oracle import SEED


@pytest.fixture
async def http_base_url(tool_server: MCPServer) -> AsyncIterator[str]:
    """Serve the tool server on a free localhost port for the duration of a test."""
    app = create_http_app(tool_server, Settings(_env_file=None))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, ws="none", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    async with asyncio.timeout(5):  # never hang the test run
        while not server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
            await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def test_tool_call_over_http(http_base_url: str, nino: Principal) -> None:
    gateway = MCPToolGateway(f"{http_base_url}/mcp", auth_secret=TEST_MCP_SECRET)

    result = await gateway.call("get_account_balance", {"currency": "GEL"}, principal=nino)

    assert isinstance(result.output, BalanceResult)
    assert all(
        isinstance(a.balance, Decimal) for a in result.output.accounts
    )  # survived JSON as exact strings
    assert len(result.output.accounts) == 2


async def test_whole_agent_turns_over_http(http_base_url: str) -> None:
    """LangGraph -> MCP over streamable HTTP -> tool server -> database: the deployed topology."""
    gateway = MCPToolGateway(f"{http_base_url}/mcp", auth_secret=TEST_MCP_SECRET)
    checking = next(a for a in SEED.accounts if a.user_id == NINO)
    bot = BankingAssistant(AgentDeps(gateway=gateway, clock=lambda: ANCHOR))

    reply = await bot.respond(user_id=NINO, message="What's my current balance?")

    assert reply.tool == "get_account_balance" and f"{checking.balance:,.2f} GEL" in reply.response
    # The principal travels as a signed token in the request metadata; over HTTP the
    # server still refuses another customer's account, even when the LLM asks for it.
    route = json.dumps(
        {"intent": "transactions", "arguments": {"account_id": str(GIORGI_CHECKING)}, "confidence": 0.9}
    )
    llm_bot = BankingAssistant(AgentDeps(gateway=gateway, llm=ScriptedLLM(route), clock=lambda: ANCHOR))
    denied = await llm_bot.respond(user_id=NINO, message=f"anything new on {GIORGI_CHECKING}?")

    assert denied.tool_error == "access_denied" and denied.tool_result is None
    assert str(GIORGI_CHECKING) not in denied.response


async def test_health_endpoint(http_base_url: str) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{http_base_url}/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "tools": 8}


async def test_foreign_host_header_is_rejected(http_base_url: str) -> None:
    """DNS-rebinding protection: a browser tricked into calling localhost sends a foreign Host."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{http_base_url}/mcp",
            json=body,
            headers={"Host": "evil.example.com", "Accept": "application/json, text/event-stream"},
        )

    assert 400 <= response.status_code < 500


async def test_server_down_is_reported_as_unavailable(nino: Principal) -> None:
    gateway = MCPToolGateway(f"http://127.0.0.1:{free_port()}/mcp", auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {}, principal=nino)

    assert caught.value.code == "unavailable"
    assert caught.value.retryable is True


async def test_slow_tool_times_out(nino: Principal) -> None:
    slow = MCPServer(name="slow")

    @slow.tool(name="get_account_balance")
    async def sleepy() -> str:
        await asyncio.sleep(5)
        return "too late"

    gateway = MCPToolGateway(slow, auth_secret=TEST_MCP_SECRET, timeout_seconds=0.2)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {}, principal=nino)

    assert caught.value.code == "timeout"
    assert caught.value.retryable is True


async def test_malformed_tool_output_is_rejected(nino: Principal) -> None:
    """If a (compromised or buggy) server returns the wrong shape, the agent must not use it."""
    liar = MCPServer(name="liar")

    @liar.tool(name="get_account_balance")
    async def wrong_shape() -> dict[str, Any]:
        return {"balance": "a lot"}

    gateway = MCPToolGateway(liar, auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {}, principal=nino)

    assert caught.value.code == "invalid_tool_output"


def test_run_wires_settings_into_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(server_module.uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    monkeypatch.setenv("MCP_PORT", "8123")
    server_module.get_settings.cache_clear()
    try:
        server_module.run()
    finally:
        server_module.get_settings.cache_clear()

    assert captured["port"] == 8123
    assert captured["host"] == "127.0.0.1"
    assert captured["log_config"] is None
    assert captured["app"] is not None


def test_gateway_from_settings() -> None:
    settings = Settings(_env_file=None, mcp_server_url="http://mcp-server:8001/mcp", tool_timeout_seconds=3)

    gateway = MCPToolGateway.from_settings(settings)

    assert gateway._server == "http://mcp-server:8001/mcp"
    assert gateway._timeout_seconds == 3


async def test_plain_text_sdk_errors_are_mapped(nino: Principal) -> None:
    """Errors raised outside our tool wrapper arrive as text, without our structured code."""
    broken = MCPServer(name="broken")

    @broken.tool(name="get_account_balance")
    async def explode() -> str:
        raise RuntimeError("boom")

    gateway = MCPToolGateway(broken, auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {}, principal=nino)

    assert caught.value.code == "tool_error"


def test_module_entry_point_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[bool] = []
    monkeypatch.setattr(server_module, "run", lambda: started.append(True))

    runpy.run_module("app.mcp", run_name="__main__")

    assert started == [True]
