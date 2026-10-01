"""Tests for the pure mock-data generator (no database involved).

These are data-quality tests: the evaluation framework computes ground truth
from this data, so the data itself must be trustworthy.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from app.core.time import BANK_TZ
from app.database.seed import (
    HISTORY_DAYS,
    PRODUCTS,
    SALARY_DAY,
    SeedData,
    generate_seed_data,
    stable_uuid,
)
from app.models.enums import AccountType, Currency, ProductType, TransactionCategory, TransactionDirection
from tests.constants import ANCHOR


@pytest.fixture(scope="module")
def data() -> SeedData:
    return generate_seed_data(anchor=ANCHOR)


def signed(direction: TransactionDirection, amount: Decimal) -> Decimal:
    return amount if direction is TransactionDirection.CREDIT else -amount


# --- volume -------------------------------------------------------------------


def test_meets_minimum_dataset_size(data: SeedData) -> None:
    assert len(data.users) >= 10
    assert len(data.accounts) >= 15
    assert len(data.transactions) >= 200
    assert len(data.products) == len(PRODUCTS)


def test_every_user_has_a_gel_checking_account(data: SeedData) -> None:
    for user in data.users:
        owned = [a for a in data.accounts if a.user_id == user.id]
        assert any(a.account_type is AccountType.CHECKING and a.currency is Currency.GEL for a in owned)


def test_required_spending_categories_are_present(data: SeedData) -> None:
    present = {t.category for t in data.transactions}
    required = {
        TransactionCategory.GROCERIES,
        TransactionCategory.RESTAURANTS,
        TransactionCategory.TRANSPORTATION,
        TransactionCategory.SHOPPING,
        TransactionCategory.UTILITIES,
        TransactionCategory.SALARY,
        TransactionCategory.SUBSCRIPTIONS,
    }
    assert required <= present


# --- determinism --------------------------------------------------------------


def snapshot(data: SeedData) -> list[tuple[object, ...]]:
    return [(t.id, t.account_id, t.amount, t.category, t.booked_at) for t in data.transactions]


def test_same_seed_and_anchor_produce_identical_data(data: SeedData) -> None:
    again = generate_seed_data(anchor=ANCHOR)

    assert snapshot(again) == snapshot(data)
    assert [a.balance for a in again.accounts] == [a.balance for a in data.accounts]


def test_different_seed_produces_different_amounts(data: SeedData) -> None:
    other = generate_seed_data(anchor=ANCHOR, seed=7)

    assert [a.balance for a in other.accounts] != [a.balance for a in data.accounts]


def test_entity_ids_are_stable_across_seeds() -> None:
    other = generate_seed_data(anchor=ANCHOR, seed=7)

    assert other.users[0].id == stable_uuid("user", 1)
    assert other.accounts[0].id == stable_uuid("account", 1, 1)


def test_naive_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        generate_seed_data(anchor=datetime(2026, 9, 28))  # noqa: DTZ001


# --- ledger integrity ---------------------------------------------------------


def test_balance_equals_sum_of_ledger(data: SeedData) -> None:
    ledger: dict[object, Decimal] = defaultdict(Decimal)
    for tx in data.transactions:
        ledger[tx.account_id] += signed(tx.direction, tx.amount)

    for account in data.accounts:
        assert account.balance == ledger[account.id], account.account_number


def test_running_balance_never_goes_negative(data: SeedData) -> None:
    by_account = defaultdict(list)
    for tx in data.transactions:
        by_account[tx.account_id].append(tx)

    for account_id, txs in by_account.items():
        running = Decimal("0")
        for tx in sorted(txs, key=lambda t: t.booked_at):
            running += signed(tx.direction, tx.amount)
            assert running >= 0, f"account {account_id} negative at {tx.booked_at}"


def test_savings_transfers_appear_on_both_sides(data: SeedData) -> None:
    for user in data.users:
        accounts = [a for a in data.accounts if a.user_id == user.id]
        checking = accounts[0]
        gel_savings = [
            a for a in accounts if a.account_type is AccountType.SAVINGS and a.currency is Currency.GEL
        ]
        if not gel_savings:
            continue
        out = sum(
            t.amount
            for t in data.transactions
            if t.account_id == checking.id and t.category is TransactionCategory.TRANSFER
        )
        into = sum(
            t.amount
            for t in data.transactions
            if t.account_id == gel_savings[0].id and t.category is TransactionCategory.TRANSFER
        )
        assert out == into > 0


def test_amounts_are_positive_with_two_decimals(data: SeedData) -> None:
    for tx in data.transactions:
        assert tx.amount > 0
        assert tx.amount == tx.amount.quantize(Decimal("0.01"))


def test_transaction_currency_matches_account(data: SeedData) -> None:
    currency = {a.id: a.currency for a in data.accounts}
    assert all(tx.currency is currency[tx.account_id] for tx in data.transactions)


def test_transactions_fall_inside_history_window(data: SeedData) -> None:
    start = ANCHOR - timedelta(days=HISTORY_DAYS)
    assert all(start <= tx.booked_at <= ANCHOR for tx in data.transactions)


def test_salary_is_paid_on_the_25th_in_bank_time(data: SeedData) -> None:
    salaries = [t for t in data.transactions if t.category is TransactionCategory.SALARY]

    assert salaries
    for tx in salaries:
        assert tx.direction is TransactionDirection.CREDIT
        assert tx.booked_at.astimezone(BANK_TZ).day == SALARY_DAY


# --- "clearly mock" guarantees --------------------------------------------------


def test_personal_data_is_obviously_fictional(data: SeedData) -> None:
    assert all(u.email.endswith("@example.com") for u in data.users)
    assert all("MOCK" in a.account_number and a.account_number[2:4] == "00" for a in data.accounts)


def test_identifiers_are_unique(data: SeedData) -> None:
    for values in (
        [u.email for u in data.users],
        [u.customer_number for u in data.users],
        [a.account_number for a in data.accounts],
        [t.id for t in data.transactions],
        [p.code for p in data.products],
    ):
        assert len(values) == len(set(values))


@pytest.mark.parametrize(
    "product", [p for p in PRODUCTS if p["product_type"] in (ProductType.LOAN, ProductType.MORTGAGE)]
)
def test_credit_products_have_complete_terms(product: dict[str, object]) -> None:
    rate, low, high = product["interest_rate_pct"], product["min_amount"], product["max_amount"]
    min_term, max_term = product["min_term_months"], product["max_term_months"]

    assert isinstance(rate, Decimal) and rate > 0
    assert isinstance(low, Decimal) and isinstance(high, Decimal) and low < high
    assert isinstance(min_term, int) and isinstance(max_term, int) and min_term <= max_term


def test_seeding_just_after_midnight_never_creates_future_transactions() -> None:
    # Edge case: seeding at 00:30 on the 1st of a month. Today's random card
    # payments (08:00-22:00) would be in the future and must be dropped.
    anchor = datetime(2026, 10, 1, 0, 30, tzinfo=BANK_TZ)

    early = generate_seed_data(anchor=anchor)

    assert all(tx.booked_at <= anchor for tx in early.transactions)
    assert len(early.transactions) >= 200
