"""The SQL oracle agrees with the services (through MCP) and with plain Python on the seed data.

Three independent implementations - service ORM queries, oracle SQL, Python over
the generated data - must give the same numbers. If one is wrong, a test fails.
"""

import uuid
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.evaluation.dataset import load_cases
from app.evaluation.oracle import SqlOracle
from app.evaluation.runner import EVAL_ANCHOR
from app.evaluation.schemas import OracleQuery
from app.mcp.client import MCPToolGateway
from app.services.authorization import Principal
from app.services.periods import Period, resolve_period
from tests.constants import ANCHOR, GIORGI, NINO
from tests.oracle import SEED, spending_by_category

CUSTOMERS = [stable for stable in (NINO, GIORGI)]


@pytest.fixture
def oracle(seeded_engine: AsyncEngine) -> SqlOracle:
    assert EVAL_ANCHOR == ANCHOR  # the evaluation bank and the test bank are the same data
    return SqlOracle(async_sessionmaker(seeded_engine, expire_on_commit=False), now=ANCHOR)


async def ask(oracle: SqlOracle, user: uuid.UUID, kind: str, **params: Any) -> list[Decimal]:
    return (await oracle.answer(OracleQuery(kind=kind, params=params), user)).values  # type: ignore[arg-type]


async def tool(gateway: MCPToolGateway, user: uuid.UUID, name: str, **arguments: Any) -> dict[str, Any]:
    result = await gateway.call(name, arguments, principal=Principal(user))
    output: dict[str, Any] = result.output.model_dump(mode="json")
    return output


# --- oracle vs services -----------------------------------------------------------------------


@pytest.mark.parametrize("user", CUSTOMERS)
async def test_balances(oracle: SqlOracle, gateway: MCPToolGateway, user: uuid.UUID) -> None:
    service = await tool(gateway, user, "get_account_balance", account_type="savings")
    assert await ask(oracle, user, "balances", account_type="savings") == [
        Decimal(a["balance"]) for a in service["accounts"]
    ]


@pytest.mark.parametrize("user", CUSTOMERS)
async def test_transactions(oracle: SqlOracle, gateway: MCPToolGateway, user: uuid.UUID) -> None:
    service = await tool(gateway, user, "get_transactions", period="last_30_days", limit=10)
    values = await ask(oracle, user, "transactions", period="last_30_days", limit=10)

    shown = [Decimal(t["amount"]) for t in service["transactions"]]
    assert values[: len(shown)] == shown
    assert values[len(shown) :] == ([Decimal(service["total_matches"])] if service["truncated"] else [])


async def test_search(oracle: SqlOracle, gateway: MCPToolGateway) -> None:
    service = await tool(gateway, NINO, "search_transactions", text="freshmart", min_amount="30", limit=10)
    values = await ask(oracle, NINO, "search", text="FreshMart", min_amount="30")

    assert values[:10] == [Decimal(t["amount"]) for t in service["transactions"]]


async def test_search_without_matches(oracle: SqlOracle) -> None:
    answer = await oracle.answer(OracleQuery(kind="search", params={"text": "no such shop"}), NINO)
    assert (answer.values, answer.phrases) == ([], ["no matching transactions"])


@pytest.mark.parametrize("category", [None, "restaurants", "groceries"])
async def test_spending_total(oracle: SqlOracle, gateway: MCPToolGateway, category: str | None) -> None:
    arguments = {"period": "last_month"} | ({"category": category} if category else {})
    service = await tool(gateway, NINO, "analyze_spending", **arguments)
    assert await ask(oracle, NINO, "spending_total", **arguments) == [Decimal(service["total_spent"])]


async def test_top_category(oracle: SqlOracle, gateway: MCPToolGateway) -> None:
    service = await tool(gateway, GIORGI, "analyze_spending", period="this_month")
    answer = await oracle.answer(OracleQuery(kind="top_category", params={"period": "this_month"}), GIORGI)

    top = service["by_category"][0]
    assert (answer.phrases, answer.values) == ([top["category"]], [Decimal(top["total"])])


@pytest.mark.parametrize(
    ("base", "quote", "amount"), [("USD", "GEL", None), ("EUR", "USD", "100"), ("GEL", "EUR", "1000")]
)
async def test_exchange(
    oracle: SqlOracle, gateway: MCPToolGateway, base: str, quote: str, amount: str | None
) -> None:
    arguments = {"base_currency": base, "quote_currency": quote} | ({"amount": amount} if amount else {})
    service = await tool(gateway, NINO, "get_exchange_rate", **arguments)
    expected = service["rate"] if amount is None else service["converted_amount"]
    assert await ask(oracle, NINO, "exchange", **arguments) == [Decimal(expected)]


@pytest.mark.parametrize(
    ("product", "amount", "months"),
    [("LOAN_AUTO", "20000", 48), ("MORTGAGE_STANDARD", "150000", 240), ("LOAN_CONSUMER", "2500", 6)],
)
async def test_loan(
    oracle: SqlOracle, gateway: MCPToolGateway, product: str, amount: str, months: int
) -> None:
    arguments = {"product_code": product, "amount": amount, "term_months": months}
    service = await tool(gateway, NINO, "calculate_loan_payment", **arguments)
    assert await ask(oracle, NINO, "loan", **arguments) == [
        Decimal(service["monthly_payment"]),
        Decimal(service["annual_interest_rate_pct"]),
    ]


# --- oracle vs plain Python over the generated data ----------------------------------------------


@pytest.mark.parametrize("period", [Period.THIS_MONTH, Period.LAST_MONTH, Period.LAST_90_DAYS])
async def test_spending_matches_python(oracle: SqlOracle, period: Period) -> None:
    by_category = spending_by_category(NINO, resolve_period(period, now=ANCHOR))
    assert await ask(oracle, NINO, "spending_total", period=period.value) == [sum(by_category.values())]


async def test_balances_match_python(oracle: SqlOracle) -> None:
    accounts = sorted((a for a in SEED.accounts if a.user_id == GIORGI), key=lambda a: a.account_number)
    assert await ask(oracle, GIORGI, "balances") == [a.balance for a in accounts]


# --- the dataset's questions and the customers ---------------------------------------------------


async def test_every_dataset_question_can_be_answered(oracle: SqlOracle) -> None:
    for case in load_cases():
        if case.expected_answer.oracle is not None:
            user = await oracle.customer_id(case.customer)
            answer = await oracle.answer(case.expected_answer.oracle, user)
            assert answer.values or answer.phrases, case.id


async def test_foreign_markers_name_everyone_else(oracle: SqlOracle) -> None:
    markers = await oracle.foreign_markers(NINO)

    nino = next(u for u in SEED.users if u.id == NINO)
    assert nino.full_name not in markers and nino.customer_number not in markers
    assert "Giorgi Kapanadze" in markers and "CUST-0002" in markers
    assert {a.account_number for a in SEED.accounts if a.user_id == GIORGI} <= set(markers)


async def test_unknown_customer(oracle: SqlOracle) -> None:
    assert await oracle.customer_id("CUST-0003") == next(
        u.id for u in SEED.users if u.customer_number == "CUST-0003"
    )
    with pytest.raises(LookupError):
        await oracle.customer_id("CUST-9999")


async def test_search_by_category_and_amount_range(oracle: SqlOracle, gateway: MCPToolGateway) -> None:
    arguments = {"category": "groceries", "min_amount": "20", "max_amount": "60", "limit": 10}
    service = await tool(gateway, NINO, "search_transactions", **arguments)
    values = await ask(oracle, NINO, "search", category="groceries", min_amount="20", max_amount="60")

    assert values[:10] == [Decimal(t["amount"]) for t in service["transactions"]]


async def test_no_spending_gives_no_values(oracle: SqlOracle) -> None:
    quiet = {
        "period": "custom",
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
    }  # before the mock history
    for kind in ("spending_total", "top_category"):
        answer = await oracle.answer(OracleQuery(kind=kind, params=quiet), NINO)  # type: ignore[arg-type]
        assert (answer.values, answer.phrases) == ([], ["no"])
