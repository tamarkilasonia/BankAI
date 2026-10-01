"""Security tests: attacking the MCP tool boundary.

Scenario: Nino (the attacker, authenticated) tries to reach Giorgi's (the
victim's) data. Every attack must fail *and* the response must not contain
any of the victim's data. Several attacks bypass our own gateway and talk to
the MCP server directly, because a real attacker would not use our client.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.core.security import PRINCIPAL_META_KEY, TOKEN_AUDIENCE, TOKEN_ISSUER, create_principal_token
from app.mcp.client import MCPToolGateway, ToolCallError
from app.mcp.schemas import TOOLS
from app.models import Account, Transaction
from app.services.authorization import Principal
from tests.constants import GIORGI, GIORGI_CHECKING, NINO, TEST_MCP_SECRET
from tests.oracle import SEED

VICTIM_ACCOUNTS = [a for a in SEED.accounts if a.user_id == GIORGI]
# Strings that would prove a leak if they appeared in any response.
VICTIM_MARKERS = [str(a.id) for a in VICTIM_ACCOUNTS] + [a.account_number for a in VICTIM_ACCOUNTS]
VICTIM_MARKERS += [str(a.balance) for a in VICTIM_ACCOUNTS] + ["Giorgi", "Kapanadze", "CUST-0002"]

FOREIGN_ACCOUNT_CALLS = [
    ("get_account_balance", {"account_id": str(GIORGI_CHECKING)}),
    ("get_account_details", {"account_id": str(GIORGI_CHECKING)}),
    ("get_transactions", {"account_id": str(GIORGI_CHECKING)}),
    ("search_transactions", {"account_id": str(GIORGI_CHECKING), "category": "groceries"}),
    ("analyze_spending", {"account_id": str(GIORGI_CHECKING)}),
]


def assert_no_victim_data(payload: object) -> None:
    text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    leaked = [marker for marker in VICTIM_MARKERS if marker in text]
    assert not leaked, f"victim data leaked: {leaked}"


async def raw_call(
    server: MCPServer, tool: str, arguments: dict[str, Any], token: str | None
) -> CallToolResult:
    """Call the server directly, bypassing MCPToolGateway and its validation."""
    meta = {PRINCIPAL_META_KEY: token} if token is not None else None
    async with Client(server) as client:
        return await client.call_tool(tool, arguments, meta=meta)  # type: ignore[arg-type]


def error_code(result: CallToolResult) -> str | None:
    content = result.structured_content or {}
    return content.get("error", {}).get("code") if result.is_error else None


# --- 1. horizontal privilege escalation ------------------------------------------


@pytest.mark.parametrize(("tool", "arguments"), FOREIGN_ACCOUNT_CALLS)
async def test_foreign_account_id_is_denied_without_leaking(
    gateway: MCPToolGateway, nino: Principal, tool: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ToolCallError) as caught:
        await gateway.call(tool, arguments, principal=nino)

    assert caught.value.code == "access_denied"
    assert_no_victim_data(caught.value.message)


async def test_foreign_and_nonexistent_accounts_look_identical(tool_server: MCPServer) -> None:
    """Otherwise the error itself would reveal which account IDs exist (enumeration)."""
    token = create_principal_token(NINO, secret=TEST_MCP_SECRET)

    foreign = await raw_call(tool_server, "get_transactions", {"account_id": str(GIORGI_CHECKING)}, token)
    missing = await raw_call(tool_server, "get_transactions", {"account_id": str(uuid.uuid4())}, token)

    assert foreign.structured_content == missing.structured_content
    assert error_code(foreign) == "access_denied"


# --- 2. identity smuggling through tool arguments ----------------------------------


@pytest.mark.parametrize("field", ["user_id", "customer_id", "owner_id", "principal_token"])
async def test_gateway_rejects_identity_arguments(
    gateway: MCPToolGateway, nino: Principal, field: str
) -> None:
    with pytest.raises(ToolCallError) as caught:
        await gateway.call("get_account_balance", {field: str(GIORGI)}, principal=nino)

    assert caught.value.code == "invalid_arguments"


@pytest.mark.parametrize("tool", ["get_account_balance", "get_transactions", "analyze_spending"])
async def test_server_never_lets_arguments_override_the_token(tool_server: MCPServer, tool: str) -> None:
    """Bypassing the gateway: an injected user_id must not change whose data is returned.

    Observed behaviour: the MCP SDK drops undeclared arguments before our code
    runs, so the call is answered strictly for the token holder. (Our gateway
    rejects such arguments loudly; see test_gateway_rejects_identity_arguments.)
    """
    attacker_token = create_principal_token(NINO, secret=TEST_MCP_SECRET)

    result = await raw_call(
        tool_server, tool, {"user_id": str(GIORGI), "customer_number": "CUST-0002"}, attacker_token
    )

    assert not result.is_error
    assert_no_victim_data(result.structured_content)
    if tool == "get_account_balance":
        owners = {a.user_id for a in SEED.accounts if str(a.id) in json.dumps(result.structured_content)}
        assert owners == {NINO}


# --- 3. token attacks ------------------------------------------------------------------


def forged_tokens() -> dict[str, str | None]:
    now = datetime.now(UTC)
    victim_claims = {"sub": str(GIORGI), "iss": TOKEN_ISSUER, "aud": TOKEN_AUDIENCE, "iat": now}
    return {
        "missing": None,
        "empty": "",
        "garbage": "not-a-token",
        "wrong secret": create_principal_token(GIORGI, secret="attacker-guessed-secret-0123456789abcdef"),
        "expired": create_principal_token(GIORGI, secret=TEST_MCP_SECRET, now=now - timedelta(hours=1)),
        "alg none": jwt.encode(
            victim_claims | {"exp": now + timedelta(minutes=5)}, key=None, algorithm="none"
        ),  # type: ignore[arg-type]
        "wrong audience": jwt.encode(
            victim_claims | {"aud": "other-service", "exp": now + timedelta(minutes=5)},
            TEST_MCP_SECRET,
            "HS256",
        ),
    }


@pytest.mark.parametrize("case", list(forged_tokens()))
async def test_invalid_tokens_get_nothing(tool_server: MCPServer, case: str) -> None:
    token = forged_tokens()[case]

    result = await raw_call(tool_server, "get_account_balance", {}, token)

    assert error_code(result) == "unauthenticated"
    assert_no_victim_data(result.structured_content)


async def test_token_in_arguments_instead_of_meta_is_not_honoured(tool_server: MCPServer) -> None:
    victim_token = create_principal_token(GIORGI, secret=TEST_MCP_SECRET)

    result = await raw_call(tool_server, "get_account_balance", {PRINCIPAL_META_KEY: victim_token}, None)

    assert result.is_error
    assert_no_victim_data(result.structured_content)


# --- 4. injection & resource exhaustion -------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    ["' OR '1'='1", "'; DROP TABLE transactions; --", "%' UNION SELECT * FROM users --", "\\\\", "%_%"],
)
async def test_sql_injection_strings_are_just_text(
    gateway: MCPToolGateway, nino: Principal, seeded_engine: AsyncEngine, payload: str
) -> None:
    result = await gateway.call(
        "search_transactions", {"text": payload, "period": "last_90_days"}, principal=nino
    )

    assert result.output.total_matches == 0  # type: ignore[attr-defined]
    async with async_sessionmaker(seeded_engine)() as session:
        assert await session.scalar(select(func.count()).select_from(Transaction)) == len(SEED.transactions)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_transactions", {"limit": 100_000}),
        ("get_transactions", {"period": "custom", "start_date": "1990-01-01", "end_date": "2026-09-28"}),
        ("search_transactions", {"text": "x" * 10_000}),
        ("analyze_spending", {"top_merchants": 1_000}),
    ],
)
async def test_oversized_requests_are_refused(
    gateway: MCPToolGateway, nino: Principal, tool: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ToolCallError) as caught:
        await gateway.call(tool, arguments, principal=nino)

    assert caught.value.code in {"invalid_arguments", "rule_violation"}


# --- 5. no tool can change data -----------------------------------------------------------


async def test_no_tool_changes_the_database(
    gateway: MCPToolGateway, nino: Principal, seeded_engine: AsyncEngine
) -> None:
    async def snapshot() -> tuple[int, list[tuple[Any, ...]]]:
        async with async_sessionmaker(seeded_engine)() as session:
            count = await session.scalar(select(func.count()).select_from(Transaction))
            balances = (await session.execute(select(Account.id, Account.balance).order_by(Account.id))).all()
            return count or 0, [tuple(row) for row in balances]

    before = await snapshot()
    sample_args: dict[str, dict[str, Any]] = {
        "search_transactions": {"text": "transfer"},
        "get_exchange_rate": {"base_currency": "USD", "quote_currency": "GEL", "amount": "5000"},
        "calculate_loan_payment": {"product_code": "LOAN_CONSUMER", "amount": "5000", "term_months": 12},
        "search_banking_policy": {"query": "card fees"},
    }
    for tool in TOOLS:
        await gateway.call(tool, sample_args.get(tool, {}), principal=nino)

    assert await snapshot() == before


def test_every_tool_is_named_as_a_read_operation() -> None:
    """Allow-list, not deny-list: a new tool named e.g. 'make_payment' fails this test."""
    read_verbs = {"get", "search", "analyze", "calculate"}
    assert all(name.split("_")[0] in read_verbs for name in TOOLS)
