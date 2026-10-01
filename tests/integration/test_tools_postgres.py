"""The MCP tools against real PostgreSQL (timestamptz, NUMERIC, ILIKE, GROUP BY)."""

from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.database.seed import seed_database
from app.mcp.client import MCPToolGateway, ToolCallError
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.services.authorization import Principal
from app.services.schemas import SpendingReport, TransactionList
from tests.constants import ANCHOR, GIORGI_CHECKING, NINO, TEST_MCP_SECRET
from tests.oracle import spending_by_category, user_transactions

pytestmark = pytest.mark.postgres


@pytest.fixture
async def pg_gateway(pg_engine: AsyncEngine) -> AsyncIterator[MCPToolGateway]:
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with factory() as session:
        await seed_database(session, anchor=ANCHOR, reset=True)
    server = build_mcp_server(
        ToolServerDeps(session_factory=factory, auth_secret=TEST_MCP_SECRET, clock=lambda: ANCHOR)
    )
    yield MCPToolGateway(server, auth_secret=TEST_MCP_SECRET)


async def test_spending_matches_oracle_on_postgres(pg_gateway: MCPToolGateway) -> None:
    result = await pg_gateway.call("analyze_spending", {"period": "last_month"}, principal=Principal(NINO))

    assert isinstance(result.output, SpendingReport)
    expected = spending_by_category(NINO, result.output.period)
    assert {c.category.value: c.total for c in result.output.by_category} == expected


async def test_period_filter_on_timestamptz(pg_gateway: MCPToolGateway) -> None:
    result = await pg_gateway.call(
        "get_transactions", {"period": "this_month", "limit": 100}, principal=Principal(NINO)
    )

    assert isinstance(result.output, TransactionList)
    period = result.output.period
    assert result.output.total_matches == len(user_transactions(NINO, period.start, period.end))


async def test_case_insensitive_search_and_literal_wildcards(pg_gateway: MCPToolGateway) -> None:
    found = await pg_gateway.call("search_transactions", {"text": "FRESHMART"}, principal=Principal(NINO))
    wildcard = await pg_gateway.call("search_transactions", {"text": "%%"}, principal=Principal(NINO))

    assert found.output.total_matches > 0  # type: ignore[attr-defined]
    assert wildcard.output.total_matches == 0  # type: ignore[attr-defined]


async def test_amount_filter_uses_exact_decimals(pg_gateway: MCPToolGateway) -> None:
    result = await pg_gateway.call(
        "search_transactions", {"min_amount": "24.99", "max_amount": "24.99"}, principal=Principal(NINO)
    )

    assert all(t.amount == Decimal("24.99") for t in result.output.transactions)  # type: ignore[attr-defined]


async def test_cross_customer_access_denied_on_postgres(pg_gateway: MCPToolGateway) -> None:
    with pytest.raises(ToolCallError, match="access_denied"):
        await pg_gateway.call(
            "get_transactions", {"account_id": str(GIORGI_CHECKING)}, principal=Principal(NINO)
        )
