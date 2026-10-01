"""Database behaviour on SQLite: schema, constraints, cascades and seeding.

Constraint tests prove the database itself protects data integrity, not just
the Python code that usually writes to it.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.database.seed import SeedError, seed_database, stable_uuid
from app.database.session import check_database, create_engine, get_db_session
from app.models import Account, ConversationLog, Transaction, User
from app.models.enums import (
    AccountStatus,
    AccountType,
    Channel,
    Currency,
    Language,
    TransactionCategory,
    TransactionDirection,
)
from tests.constants import ANCHOR

USER_1 = stable_uuid("user", 1)
ACCOUNT_1 = stable_uuid("account", 1, 1)


async def count(session: AsyncSession, model: type[object]) -> int:
    return await session.scalar(select(func.count()).select_from(model)) or 0


def new_transaction(**overrides: object) -> Transaction:
    values: dict[str, object] = {
        "account_id": ACCOUNT_1,
        "direction": TransactionDirection.DEBIT,
        "amount": Decimal("10.00"),
        "currency": Currency.GEL,
        "category": TransactionCategory.GROCERIES,
        "merchant": "FreshMart",
        "description": "Card payment - FreshMart",
        "booked_at": ANCHOR,
    }
    return Transaction(**(values | overrides))


# --- seeding ------------------------------------------------------------------


async def test_seed_persists_the_full_dataset(seeded_session: AsyncSession) -> None:
    assert await count(seeded_session, User) == 10
    assert await count(seeded_session, Account) >= 15
    assert await count(seeded_session, Transaction) >= 200


async def test_seed_refuses_to_run_twice_without_reset(seeded_session: AsyncSession) -> None:
    with pytest.raises(SeedError, match="--reset"):
        await seed_database(seeded_session, anchor=ANCHOR)


async def test_seed_with_reset_replaces_data_instead_of_duplicating(seeded_session: AsyncSession) -> None:
    before = await count(seeded_session, Transaction)

    summary = await seed_database(seeded_session, anchor=ANCHOR, reset=True)

    assert await count(seeded_session, Transaction) == before == summary.transactions
    assert await count(seeded_session, User) == summary.users == 10


async def test_reset_keeps_conversation_logs_but_detaches_users(seeded_session: AsyncSession) -> None:
    seeded_session.add(
        ConversationLog(
            request_id="r1", conversation_id="c1", user_id=USER_1, user_input="What's my balance?"
        )
    )
    await seeded_session.commit()

    await seed_database(seeded_session, anchor=ANCHOR, reset=True)

    log = await seeded_session.scalar(select(ConversationLog).execution_options(populate_existing=True))
    assert log is not None
    assert log.user_id is None  # ON DELETE SET NULL


async def test_enums_are_stored_as_readable_values(seeded_session: AsyncSession) -> None:
    raw = await seeded_session.execute(
        text("SELECT account_type, currency, status FROM accounts WHERE id = :id"), {"id": ACCOUNT_1.hex}
    )
    assert raw.one() == ("checking", "GEL", "active")


# --- constraints --------------------------------------------------------------


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-5.00")])
async def test_transaction_amount_must_be_positive(seeded_session: AsyncSession, amount: Decimal) -> None:
    seeded_session.add(new_transaction(amount=amount))
    with pytest.raises(IntegrityError, match="amount_positive"):
        await seeded_session.commit()


async def test_account_balance_cannot_be_negative(seeded_session: AsyncSession) -> None:
    account = await seeded_session.get(Account, ACCOUNT_1)
    assert account is not None
    account.balance = Decimal("-0.01")
    with pytest.raises(IntegrityError, match="balance_non_negative"):
        await seeded_session.commit()


async def test_foreign_keys_are_enforced(seeded_session: AsyncSession) -> None:
    seeded_session.add(new_transaction(account_id=uuid.uuid4()))
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        await seeded_session.commit()


async def test_database_rejects_unknown_enum_value_from_raw_sql(seeded_session: AsyncSession) -> None:
    with pytest.raises(IntegrityError, match="CHECK"):
        await seeded_session.execute(
            text("UPDATE transactions SET category = 'gambling' WHERE account_id = :id"),
            {"id": ACCOUNT_1.hex},
        )


async def test_python_rejects_unknown_enum_value(seeded_session: AsyncSession) -> None:
    seeded_session.add(new_transaction(category="gambling"))
    with pytest.raises(StatementError, match="gambling"):
        await seeded_session.commit()


@pytest.mark.parametrize("field", ["email", "customer_number"])
async def test_user_identifiers_are_unique(seeded_session: AsyncSession, field: str) -> None:
    existing = await seeded_session.get(User, USER_1)
    assert existing is not None
    values = {"customer_number": "CUST-9999", "full_name": "Copy", "email": "copy@example.com"}
    values[field] = getattr(existing, field)
    seeded_session.add(User(**values))
    with pytest.raises(IntegrityError, match="UNIQUE"):
        await seeded_session.commit()


async def test_deleting_a_user_cascades_to_accounts_and_transactions(seeded_session: AsyncSession) -> None:
    user = await seeded_session.get(User, USER_1)
    assert user is not None
    account_ids = list(await seeded_session.scalars(select(Account.id).where(Account.user_id == USER_1)))
    assert account_ids

    await seeded_session.delete(user)
    await seeded_session.commit()

    orphans = await seeded_session.scalar(
        select(func.count()).select_from(Transaction).where(Transaction.account_id.in_(account_ids))
    )
    assert orphans == 0
    assert await seeded_session.get(Account, ACCOUNT_1) is None


async def test_defaults_are_applied(session: AsyncSession) -> None:
    user = User(customer_number="CUST-0100", full_name="Test User", email="test.user@example.com")
    session.add(user)
    await session.flush()
    account = Account(
        user_id=user.id,
        account_number="GE00MOCK00010000000001",
        name="Checking GEL",
        account_type=AccountType.CHECKING,
        currency=Currency.GEL,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    log = ConversationLog(request_id="r", conversation_id="c", user_input="hi")
    session.add_all([account, log])
    await session.commit()

    assert account.balance == Decimal("0.00")
    assert account.status is AccountStatus.ACTIVE
    assert user.preferred_language is Language.ENGLISH
    assert log.channel is Channel.TEXT
    assert log.safety_flags == []
    assert log.safety_status is None  # unknown until the safety layer has run
    assert user.created_at is not None


# --- session helpers ----------------------------------------------------------


async def test_check_database_reports_reachable_database(engine: AsyncEngine) -> None:
    assert await check_database(engine) is True


async def test_check_database_reports_unreachable_database() -> None:
    broken = create_engine("sqlite+aiosqlite:////nonexistent-dir/bankai.db")
    try:
        assert await check_database(broken) is False
    finally:
        await broken.dispose()


async def test_check_database_times_out_instead_of_hanging() -> None:
    # 10.255.255.1 is a non-routable address: connecting would hang without a timeout.
    unreachable = create_engine("postgresql+asyncpg://u:p@10.255.255.1:5432/db")
    try:
        assert await check_database(unreachable, timeout_seconds=0.2) is False
    finally:
        await unreachable.dispose()


@pytest.mark.usefixtures("reset_app_state")
async def test_get_db_session_yields_a_working_session() -> None:
    sessions: AsyncIterator[AsyncSession] = get_db_session()
    session = await anext(sessions)
    assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
    await sessions.aclose()
