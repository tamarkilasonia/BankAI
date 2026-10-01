"""Transaction listing/search and spending analysis, checked against the Python oracle."""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import Currency, TransactionCategory, TransactionDirection
from app.services.authorization import Principal
from app.services.errors import AccessDeniedError, BusinessRuleError
from app.services.periods import Period, resolve_period
from app.services.spending import analyze_spending
from app.services.transactions import escape_like, get_transactions, search_transactions
from tests.constants import ANCHOR, GIORGI_CHECKING, NINO, NINO_CHECKING, NINO_SAVINGS_USD
from tests.oracle import SEED, spending_by_category, user_transactions

LAST_90 = resolve_period(Period.LAST_90_DAYS, now=ANCHOR)
LAST_MONTH = resolve_period(Period.LAST_MONTH, now=ANCHOR)
NINO_P = Principal(NINO)


# --- get_transactions ---------------------------------------------------------


async def test_returns_newest_first_and_only_own_transactions(seeded_session: AsyncSession) -> None:
    result = await get_transactions(seeded_session, NINO_P, date_range=LAST_90, limit=100)

    owned = {a.id for a in SEED.accounts if a.user_id == NINO}
    assert all(t.account_id in owned for t in result.transactions)
    times = [t.booked_at for t in result.transactions]
    assert times == sorted(times, reverse=True)


async def test_total_matches_and_truncation(seeded_session: AsyncSession) -> None:
    expected_total = len(user_transactions(NINO, LAST_MONTH.start, LAST_MONTH.end))

    result = await get_transactions(seeded_session, NINO_P, date_range=LAST_MONTH, limit=5)

    assert result.total_matches == expected_total > 5
    assert len(result.transactions) == 5
    assert result.truncated is True


async def test_period_boundaries_are_respected(seeded_session: AsyncSession) -> None:
    result = await get_transactions(seeded_session, NINO_P, date_range=LAST_MONTH, limit=100)

    assert all(LAST_MONTH.start <= t.booked_at < LAST_MONTH.end for t in result.transactions)


@pytest.mark.parametrize(
    ("filters", "check"),
    [
        ({"category": TransactionCategory.SALARY}, lambda t: t.category is TransactionCategory.SALARY),
        ({"direction": TransactionDirection.CREDIT}, lambda t: t.direction is TransactionDirection.CREDIT),
        ({"account_id": NINO_SAVINGS_USD}, lambda t: t.currency is Currency.USD),
    ],
)
async def test_filters(seeded_session: AsyncSession, filters: dict[str, object], check: object) -> None:
    result = await get_transactions(seeded_session, NINO_P, date_range=LAST_90, limit=100, **filters)  # type: ignore[arg-type]

    assert result.transactions
    assert all(check(t) for t in result.transactions)  # type: ignore[operator]


async def test_foreign_account_is_denied(seeded_session: AsyncSession) -> None:
    with pytest.raises(AccessDeniedError):
        await get_transactions(seeded_session, NINO_P, date_range=LAST_90, account_id=GIORGI_CHECKING)


@pytest.mark.parametrize("limit", [0, 101])
async def test_limit_is_bounded(seeded_session: AsyncSession, limit: int) -> None:
    with pytest.raises(BusinessRuleError, match="limit"):
        await get_transactions(seeded_session, NINO_P, date_range=LAST_90, limit=limit)


# --- search_transactions ------------------------------------------------------


async def test_text_search_is_case_insensitive_over_merchant_and_description(
    seeded_session: AsyncSession,
) -> None:
    result = await search_transactions(
        seeded_session, NINO_P, date_range=LAST_90, text="freshmart", limit=100
    )

    expected = [
        t for t in user_transactions(NINO, LAST_90.start, LAST_90.end) if "freshmart" in t.merchant.lower()
    ]
    assert result.total_matches == len(expected) > 0
    assert all("FreshMart" in t.merchant for t in result.transactions)


@pytest.mark.parametrize("wildcard", ["%", "%%", "_", "__"])
async def test_like_wildcards_are_literal(seeded_session: AsyncSession, wildcard: str) -> None:
    result = await search_transactions(seeded_session, NINO_P, date_range=LAST_90, text=wildcard)

    assert result.total_matches == 0  # no merchant contains a literal % or _


def test_escape_like() -> None:
    assert escape_like(r"50%_off\x") == r"50\%\_off\\x"


async def test_amount_range(seeded_session: AsyncSession) -> None:
    result = await search_transactions(
        seeded_session,
        NINO_P,
        date_range=LAST_90,
        min_amount=Decimal("100"),
        max_amount=Decimal("150"),
        limit=100,
    )

    assert result.transactions
    assert all(Decimal("100") <= t.amount <= Decimal("150") for t in result.transactions)


async def test_category_only_search(seeded_session: AsyncSession) -> None:
    result = await search_transactions(
        seeded_session, NINO_P, date_range=LAST_90, category=TransactionCategory.UTILITIES, limit=100
    )

    assert result.transactions
    assert {t.category for t in result.transactions} == {TransactionCategory.UTILITIES}


async def test_search_needs_a_criterion(seeded_session: AsyncSession) -> None:
    with pytest.raises(BusinessRuleError, match="at least one"):
        await search_transactions(seeded_session, NINO_P, date_range=LAST_90)


async def test_search_rejects_inverted_amount_range(seeded_session: AsyncSession) -> None:
    with pytest.raises(BusinessRuleError, match="min_amount"):
        await search_transactions(
            seeded_session, NINO_P, date_range=LAST_90, min_amount=Decimal("10"), max_amount=Decimal("5")
        )


# --- analyze_spending ---------------------------------------------------------


async def test_spending_matches_oracle(seeded_session: AsyncSession) -> None:
    report = await analyze_spending(seeded_session, NINO_P, date_range=LAST_MONTH)

    expected = spending_by_category(NINO, LAST_MONTH)
    assert {c.category.value: c.total for c in report.by_category} == expected
    assert report.total_spent == sum(expected.values())
    assert report.transaction_count == sum(c.transaction_count for c in report.by_category)


async def test_spending_excludes_transfers_and_income(seeded_session: AsyncSession) -> None:
    report = await analyze_spending(seeded_session, NINO_P, date_range=LAST_90)

    excluded = {TransactionCategory.TRANSFER, TransactionCategory.SALARY, TransactionCategory.DEPOSIT}
    assert not {c.category for c in report.by_category} & excluded


async def test_spending_categories_are_sorted_and_shares_add_up(seeded_session: AsyncSession) -> None:
    report = await analyze_spending(seeded_session, NINO_P, date_range=LAST_90)

    totals = [c.total for c in report.by_category]
    assert totals == sorted(totals, reverse=True)
    assert abs(sum(c.share_pct for c in report.by_category) - 100) <= Decimal("0.5")  # rounding
    assert len(report.top_merchants) == 5


async def test_single_category_spending(seeded_session: AsyncSession) -> None:
    report = await analyze_spending(
        seeded_session, NINO_P, date_range=LAST_MONTH, category=TransactionCategory.RESTAURANTS
    )

    assert [c.category for c in report.by_category] == [TransactionCategory.RESTAURANTS]
    assert report.total_spent == spending_by_category(NINO, LAST_MONTH)["restaurants"]
    assert report.by_category[0].share_pct == Decimal("100.0")


async def test_spending_in_foreign_currency_is_separate(seeded_session: AsyncSession) -> None:
    gel = await analyze_spending(seeded_session, NINO_P, date_range=LAST_90, currency=Currency.GEL)
    usd = await analyze_spending(seeded_session, NINO_P, date_range=LAST_90, currency=Currency.USD)

    assert gel.total_spent > 0
    assert usd.total_spent == 0  # Nino's USD account only receives deposits
    assert usd.by_category == []


async def test_spending_on_one_account(seeded_session: AsyncSession) -> None:
    report = await analyze_spending(seeded_session, NINO_P, date_range=LAST_MONTH, account_id=NINO_CHECKING)
    assert report.total_spent == sum(spending_by_category(NINO, LAST_MONTH).values())


async def test_non_spending_category_is_rejected(seeded_session: AsyncSession) -> None:
    with pytest.raises(BusinessRuleError, match="not a spending category"):
        await analyze_spending(
            seeded_session, NINO_P, date_range=LAST_MONTH, category=TransactionCategory.SALARY
        )


async def test_spending_on_foreign_account_is_denied(seeded_session: AsyncSession) -> None:
    with pytest.raises(AccessDeniedError):
        await analyze_spending(seeded_session, NINO_P, date_range=LAST_MONTH, account_id=GIORGI_CHECKING)
