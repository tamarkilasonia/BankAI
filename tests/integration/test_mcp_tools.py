"""Agent gateway -> MCP protocol -> tool server -> services -> database.

The server runs in-process (no network), but every call still goes through
the real MCP client and server: tool discovery, argument schemas, ``_meta``
transport of the principal token, structured results and structured errors.
"""

import json
import logging
import uuid
from decimal import Decimal
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.core.logging import JsonFormatter, request_id_var
from app.core.security import PRINCIPAL_META_KEY, create_principal_token
from app.mcp import server as server_module
from app.mcp.client import MCPToolGateway, ToolCallError
from app.mcp.schemas import TOOLS
from app.mcp.server import ToolServerDeps
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.retriever import Retriever
from app.rag.schemas import PolicySearchResult
from app.rag.store import InMemoryVectorStore
from app.services.authorization import Principal
from app.services.schemas import (
    BalanceResult,
    ExchangeRateQuote,
    LoanPaymentQuote,
    SpendingReport,
    TransactionList,
)
from tests.constants import GIORGI_CHECKING, NINO, NINO_CHECKING, TEST_MCP_SECRET
from tests.oracle import SEED, spending_by_category

# --- discovery ----------------------------------------------------------------


async def test_server_publishes_every_tool_with_schemas(tool_server: MCPServer) -> None:
    async with Client(tool_server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == set(TOOLS)
    for name, tool in tools.items():
        assert tool.description == TOOLS[name].description
        assert tool.output_schema is not None
        assert tool.annotations is not None and tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False


async def test_published_schema_carries_the_input_model_constraints(tool_server: MCPServer) -> None:
    async with Client(tool_server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    limit = tools["get_transactions"].input_schema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 100, 20)
    assert tools["calculate_loan_payment"].input_schema["required"] == [
        "product_code",
        "amount",
        "term_months",
    ]


# --- happy paths through the gateway --------------------------------------------


async def test_balance(gateway: MCPToolGateway, nino: Principal) -> None:
    result = await gateway.call("get_account_balance", {}, principal=nino)

    assert isinstance(result.output, BalanceResult)
    expected = {a.id: a.balance for a in SEED.accounts if a.user_id == NINO}
    assert {a.account_id: a.balance for a in result.output.accounts} == expected
    assert result.duration_ms > 0


async def test_transactions_with_validated_arguments_recorded(
    gateway: MCPToolGateway, nino: Principal
) -> None:
    result = await gateway.call(
        "get_transactions",
        {"account_id": str(NINO_CHECKING), "limit": 3, "period": "last_month"},
        principal=nino,
    )

    assert isinstance(result.output, TransactionList)
    assert len(result.output.transactions) == 3
    # The exact arguments sent are kept - the evaluation framework scores them.
    assert result.arguments == {"account_id": str(NINO_CHECKING), "limit": 3, "period": "last_month"}


async def test_spending_round_trips_exact_decimals(gateway: MCPToolGateway, nino: Principal) -> None:
    result = await gateway.call("analyze_spending", {"period": "last_month"}, principal=nino)

    assert isinstance(result.output, SpendingReport)
    expected = spending_by_category(NINO, result.output.period)
    assert {c.category.value: c.total for c in result.output.by_category} == expected
    assert all(isinstance(c.total, Decimal) for c in result.output.by_category)


async def test_exchange_rate_and_loan_tools(gateway: MCPToolGateway, nino: Principal) -> None:
    fx = await gateway.call(
        "get_exchange_rate", {"base_currency": "EUR", "quote_currency": "GEL", "amount": "50"}, principal=nino
    )
    loan = await gateway.call(
        "calculate_loan_payment",
        {"product_code": "LOAN_AUTO", "amount": 20000, "term_months": 48},
        principal=nino,
    )

    assert isinstance(fx.output, ExchangeRateQuote) and fx.output.converted_amount == Decimal("159.10")
    assert isinstance(loan.output, LoanPaymentQuote) and loan.output.annual_interest_rate_pct == Decimal(
        "11.90"
    )


# --- structured errors ----------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "code"),
    [
        ("get_transactions", {"account_id": str(GIORGI_CHECKING)}, "access_denied"),
        (
            "calculate_loan_payment",
            {"product_code": "LOAN_AUTO", "amount": 100, "term_months": 24},
            "rule_violation",
        ),
        (
            "calculate_loan_payment",
            {"product_code": "LOAN_PAYDAY", "amount": 100, "term_months": 24},
            "not_found",
        ),
        ("search_transactions", {}, "invalid_arguments"),
        ("transfer_money", {"amount": 5000}, "unknown_tool"),
    ],
)
async def test_failures_become_coded_tool_errors(
    gateway: MCPToolGateway, nino: Principal, tool: str, arguments: dict[str, object], code: str
) -> None:
    with pytest.raises(ToolCallError) as caught:
        await gateway.call(tool, arguments, principal=nino)

    assert caught.value.code == code
    assert caught.value.tool == tool
    assert caught.value.retryable is False


async def test_server_validates_even_when_the_gateway_is_bypassed(tool_server: MCPServer) -> None:
    """A different MCP client may skip client-side validation; the server must not rely on it."""
    token = create_principal_token(NINO, secret=TEST_MCP_SECRET)
    async with Client(tool_server) as client:
        result = await client.call_tool(
            "search_transactions", {"min_amount": "50", "max_amount": "10"}, meta={PRINCIPAL_META_KEY: token}
        )

    assert result.is_error
    error = (result.structured_content or {})["error"]
    assert error["code"] == "invalid_arguments"
    assert "min_amount must not be greater than max_amount" in error["message"]


async def test_database_failure_is_reported_as_unavailable(
    gateway: MCPToolGateway, nino: Principal, seeded_engine: AsyncEngine
) -> None:
    async with seeded_engine.begin() as conn:
        await conn.execute(text("DROP TABLE transactions"))  # simulate a broken database

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_transactions", {}, principal=nino)

    assert caught.value.code == "unavailable"
    assert caught.value.retryable is True
    assert "transactions" not in caught.value.message  # no SQL details leak


async def test_unknown_customer_token_gets_no_data(gateway: MCPToolGateway) -> None:
    stranger = Principal(uuid.uuid4())

    balance = await gateway.call("get_account_balance", {}, principal=stranger)
    assert balance.output.accounts == []  # type: ignore[attr-defined]
    with pytest.raises(ToolCallError, match="access_denied"):
        await gateway.call("get_account_details", {}, principal=stranger)


async def test_unexpected_bug_returns_internal_error_without_details(
    monkeypatch: pytest.MonkeyPatch, seeded_engine: AsyncEngine, nino: Principal
) -> None:
    async def buggy_handler(*_: object) -> BalanceResult:
        raise ZeroDivisionError("secret internal detail")

    monkeypatch.setitem(server_module.HANDLERS, "get_account_balance", buggy_handler)
    deps = ToolServerDeps(session_factory=async_sessionmaker(seeded_engine), auth_secret=TEST_MCP_SECRET)
    gateway = MCPToolGateway(server_module.build_mcp_server(deps), auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {}, principal=nino)

    assert caught.value.code == "internal_error"
    assert "secret internal detail" not in caught.value.message
    assert "ZeroDivisionError" not in caught.value.message


class _CapturingHandler(logging.Handler):
    """Formats each record at emit time, i.e. inside the server's request context."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


async def test_request_id_is_traced_from_gateway_into_the_tool_server(
    gateway: MCPToolGateway, nino: Principal
) -> None:
    handler = _CapturingHandler()
    server_logger = logging.getLogger("app.mcp.server")
    server_logger.addHandler(handler)
    previous_level = server_logger.level
    server_logger.setLevel(logging.INFO)
    token = request_id_var.set("req-trace-42")
    try:
        await gateway.call("get_account_balance", {}, principal=nino)
    finally:
        request_id_var.reset(token)
        server_logger.removeHandler(handler)
        server_logger.setLevel(previous_level)

    completed = [line for line in handler.lines if line["event"] == "mcp.tool.completed"]
    assert completed and completed[0]["request_id"] == "req-trace-42"


# --- search_banking_policy (RAG) ----------------------------------------------------------


async def test_policy_search_returns_passages_with_scores(gateway: MCPToolGateway, nino: Principal) -> None:
    result = await gateway.call(
        "search_banking_policy", {"query": "How do I block a lost card?"}, principal=nino
    )

    output = result.output
    assert isinstance(output, PolicySearchResult)
    assert output.sufficient and output.embedding_model == "hashing-v1-768"
    assert output.passages[0].source == "cards.md"
    assert output.passages[0].score >= output.passages[-1].score


async def test_policy_search_without_a_retriever_is_unavailable(
    seeded_engine: AsyncEngine, nino: Principal
) -> None:
    deps = ToolServerDeps(session_factory=async_sessionmaker(seeded_engine), auth_secret=TEST_MCP_SECRET)
    gateway = MCPToolGateway(server_module.build_mcp_server(deps), auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("search_banking_policy", {"query": "card fees"}, principal=nino)

    assert (caught.value.code, caught.value.retryable) == ("unavailable", True)


async def test_policy_search_with_embedding_server_down_is_unavailable(
    seeded_engine: AsyncEngine, nino: Principal
) -> None:
    class BrokenEmbedder(HashingEmbedder):
        async def embed_query(self, text: str) -> list[float]:
            raise EmbeddingError("the embedding model is unreachable")

    retriever = Retriever(
        embedder=BrokenEmbedder(), store=InMemoryVectorStore(), known_words=set(), min_score=0.1
    )
    deps = ToolServerDeps(
        session_factory=async_sessionmaker(seeded_engine), auth_secret=TEST_MCP_SECRET, retriever=retriever
    )
    gateway = MCPToolGateway(server_module.build_mcp_server(deps), auth_secret=TEST_MCP_SECRET)

    with pytest.raises(ToolCallError) as caught:
        await gateway.call("search_banking_policy", {"query": "card fees"}, principal=nino)

    assert caught.value.code == "unavailable"
    assert "unreachable" not in caught.value.message  # no internals
