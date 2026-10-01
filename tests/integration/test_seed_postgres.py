"""The mock dataset in real PostgreSQL, verified with SQL.

These checks are written in plain SQL on purpose: the evaluation framework
(phase 6) computes ground truth with independent SQL queries, so SQL and the
Python generator must agree exactly.
"""

from collections import defaultdict
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.seed import generate_seed_data, seed_database, stable_uuid
from app.models.enums import TransactionCategory, TransactionDirection
from tests.constants import ANCHOR

pytestmark = pytest.mark.postgres

USER_1 = stable_uuid("user", 1)


@pytest.fixture
async def seeded(pg_session: AsyncSession) -> AsyncSession:
    await seed_database(pg_session, anchor=ANCHOR, reset=True)
    return pg_session


async def test_dataset_size(seeded: AsyncSession) -> None:
    row = (
        await seeded.execute(
            text(
                "SELECT (SELECT count(*) FROM users), (SELECT count(*) FROM accounts),"
                " (SELECT count(*) FROM transactions), (SELECT count(*) FROM banking_products)"
            )
        )
    ).one()
    users, accounts, transactions, products = row

    assert users == 10
    assert accounts >= 15
    assert transactions >= 200
    assert products > 0


async def test_every_balance_equals_its_ledger(seeded: AsyncSession) -> None:
    mismatches = (
        await seeded.execute(
            text(
                """
                SELECT a.account_number, a.balance,
                       SUM(CASE WHEN t.direction = 'credit' THEN t.amount ELSE -t.amount END) AS ledger
                FROM accounts a
                JOIN transactions t ON t.account_id = a.id
                GROUP BY a.id, a.account_number, a.balance
                HAVING a.balance <> SUM(CASE WHEN t.direction = 'credit' THEN t.amount ELSE -t.amount END)
                """
            )
        )
    ).all()

    assert mismatches == []


async def test_money_is_exact_decimal_not_float(seeded: AsyncSession) -> None:
    total = (await seeded.execute(text("SELECT SUM(amount) FROM transactions"))).scalar_one()
    expected = sum((tx.amount for tx in generate_seed_data(anchor=ANCHOR).transactions), Decimal("0"))

    assert isinstance(total, Decimal)
    assert total == expected  # exact equality - no floating point drift


async def test_sql_spending_breakdown_matches_generator(seeded: AsyncSession) -> None:
    rows = (
        await seeded.execute(
            text(
                """
                SELECT t.category, SUM(t.amount)
                FROM transactions t
                JOIN accounts a ON a.id = t.account_id
                WHERE a.user_id = :user_id AND t.direction = 'debit' AND t.currency = 'GEL'
                GROUP BY t.category
                """
            ),
            {"user_id": USER_1},
        )
    ).all()

    data = generate_seed_data(anchor=ANCHOR)
    owned = {a.id for a in data.accounts if a.user_id == USER_1}
    expected: dict[str, Decimal] = defaultdict(Decimal)
    for tx in data.transactions:
        if tx.account_id in owned and tx.direction is TransactionDirection.DEBIT and tx.currency == "GEL":
            expected[tx.category.value] += tx.amount

    assert dict(rows) == dict(expected)
    assert TransactionCategory.RESTAURANTS.value in expected


async def test_timestamps_come_back_timezone_aware(seeded: AsyncSession) -> None:
    booked_at = (await seeded.execute(text("SELECT MAX(booked_at) FROM transactions"))).scalar_one()

    assert booked_at.tzinfo is not None
    assert booked_at <= ANCHOR


@pytest.mark.parametrize(
    ("statement", "constraint"),
    [
        ("UPDATE transactions SET category = 'gambling'", "ck_transactions_transactioncategory"),
        ("UPDATE transactions SET amount = 0", "ck_transactions_amount_positive"),
        ("UPDATE accounts SET balance = -1", "ck_accounts_balance_non_negative"),
        ("UPDATE accounts SET currency = 'BTC'", "ck_accounts_currency"),
    ],
)
async def test_postgres_enforces_integrity_rules(
    seeded: AsyncSession, statement: str, constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint):
        await seeded.execute(text(statement))
    await seeded.rollback()
