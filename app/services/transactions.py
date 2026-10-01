"""Transaction listing and search.

Every query starts from ``_owned_transactions``, which joins on the account
owner - so even a query built with the wrong filters can only ever see the
principal's own transactions.
"""

import uuid
from decimal import Decimal

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import as_utc
from app.models import Account, Transaction
from app.models.enums import TransactionCategory, TransactionDirection
from app.services.authorization import Principal, get_owned_account
from app.services.errors import BusinessRuleError
from app.services.periods import DateRange
from app.services.schemas import TransactionItem, TransactionList

MAX_LIMIT = 100


def escape_like(term: str) -> str:
    """Treat ``%`` and ``_`` in user input literally, not as SQL LIKE wildcards.

    Without this, searching for "%" would match every transaction. (SQL
    injection itself is impossible here: values are always bound parameters.)
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _owned_transactions(
    session: AsyncSession, principal: Principal, date_range: DateRange, account_id: uuid.UUID | None
) -> Select[Transaction]:
    stmt = (
        select(Transaction)
        .join(Account, Transaction.account_id == Account.id)
        .where(Account.user_id == principal.user_id)
        .where(Transaction.booked_at >= date_range.start, Transaction.booked_at < date_range.end)
    )
    if account_id is not None:
        await get_owned_account(session, principal, account_id)  # explicit, clear error
        stmt = stmt.where(Transaction.account_id == account_id)
    return stmt


async def _execute(
    session: AsyncSession, stmt: Select[Transaction], date_range: DateRange, limit: int
) -> TransactionList:
    if not 1 <= limit <= MAX_LIMIT:
        raise BusinessRuleError(f"limit must be between 1 and {MAX_LIMIT}.")
    total = await session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = await session.scalars(stmt.order_by(Transaction.booked_at.desc(), Transaction.id).limit(limit))
    items = [
        TransactionItem(
            transaction_id=tx.id,
            account_id=tx.account_id,
            booked_at=as_utc(tx.booked_at),
            direction=tx.direction,
            amount=tx.amount,
            currency=tx.currency,
            category=tx.category,
            merchant=tx.merchant,
            description=tx.description,
        )
        for tx in rows
    ]
    return TransactionList(
        period=date_range, transactions=items, total_matches=total, truncated=total > len(items)
    )


async def get_transactions(
    session: AsyncSession,
    principal: Principal,
    *,
    date_range: DateRange,
    account_id: uuid.UUID | None = None,
    category: TransactionCategory | None = None,
    direction: TransactionDirection | None = None,
    limit: int = 20,
) -> TransactionList:
    """The most recent transactions in a period, newest first."""
    stmt = await _owned_transactions(session, principal, date_range, account_id)
    if category is not None:
        stmt = stmt.where(Transaction.category == category)
    if direction is not None:
        stmt = stmt.where(Transaction.direction == direction)
    return await _execute(session, stmt, date_range, limit)


async def search_transactions(
    session: AsyncSession,
    principal: Principal,
    *,
    date_range: DateRange,
    text: str | None = None,
    min_amount: Decimal | None = None,
    max_amount: Decimal | None = None,
    category: TransactionCategory | None = None,
    account_id: uuid.UUID | None = None,
    limit: int = 20,
) -> TransactionList:
    """Find transactions by merchant/description text and/or amount range."""
    if text is None and min_amount is None and max_amount is None and category is None:
        raise BusinessRuleError("Give at least one search criterion: text, amount range or category.")
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise BusinessRuleError("min_amount must not be greater than max_amount.")

    stmt = await _owned_transactions(session, principal, date_range, account_id)
    if text is not None:
        pattern = f"%{escape_like(text.strip())}%"
        stmt = stmt.where(
            or_(
                Transaction.merchant.ilike(pattern, escape="\\"),
                Transaction.description.ilike(pattern, escape="\\"),
            )
        )
    if min_amount is not None:
        stmt = stmt.where(Transaction.amount >= min_amount)
    if max_amount is not None:
        stmt = stmt.where(Transaction.amount <= max_amount)
    if category is not None:
        stmt = stmt.where(Transaction.category == category)
    return await _execute(session, stmt, date_range, limit)
