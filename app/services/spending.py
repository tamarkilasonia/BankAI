"""Spending analysis: where did the money go?

"Spending" means card payments and bills (debits in a spending category).
Transfers between the customer's own accounts and incoming money are not
spending, and amounts in different currencies are never summed together.
The aggregation runs in SQL (GROUP BY), not in Python, so it scales with data.
"""

import uuid
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy import ColumnElement, Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.models import Account, Transaction
from app.models.enums import SPENDING_CATEGORIES, Currency, TransactionCategory, TransactionDirection
from app.services.authorization import Principal, get_owned_account
from app.services.errors import BusinessRuleError
from app.services.periods import DateRange
from app.services.schemas import CategorySpending, MerchantSpending, SpendingReport

_CENT = Decimal("0.01")
_TENTH = Decimal("0.1")


def _money(value: Any) -> Decimal:
    return Decimal(value or 0).quantize(_CENT, rounding=ROUND_HALF_UP)


async def analyze_spending(
    session: AsyncSession,
    principal: Principal,
    *,
    date_range: DateRange,
    currency: Currency = Currency.GEL,
    category: TransactionCategory | None = None,
    account_id: uuid.UUID | None = None,
    top_merchants: int = 5,
) -> SpendingReport:
    if category is not None and category not in SPENDING_CATEGORIES:
        allowed = ", ".join(sorted(c.value for c in SPENDING_CATEGORIES))
        raise BusinessRuleError(f"'{category.value}' is not a spending category. Use one of: {allowed}.")
    if account_id is not None:
        await get_owned_account(session, principal, account_id)

    categories = [category] if category is not None else sorted(SPENDING_CATEGORIES)
    conditions: list[ColumnElement[bool]] = [
        Account.user_id == principal.user_id,
        Transaction.direction == TransactionDirection.DEBIT,
        Transaction.currency == currency,
        Transaction.category.in_(categories),
        Transaction.booked_at >= date_range.start,
        Transaction.booked_at < date_range.end,
    ]
    if account_id is not None:
        conditions.append(Transaction.account_id == account_id)

    def grouped_by(column: InstrumentedAttribute[Any]) -> Select[Any]:
        """SELECT <column>, SUM(amount) AS total, COUNT(*) AS n ... GROUP BY <column>, biggest first."""
        return (
            select(column, func.sum(Transaction.amount).label("total"), func.count(Transaction.id).label("n"))
            .join(Account, Transaction.account_id == Account.id)
            .where(*conditions)
            .group_by(column)
            .order_by(desc("total"), column)
        )

    category_rows = (await session.execute(grouped_by(Transaction.category))).all()
    merchant_rows = (await session.execute(grouped_by(Transaction.merchant).limit(top_merchants))).all()

    total_spent = sum((_money(row.total) for row in category_rows), Decimal("0.00"))
    by_category = [
        CategorySpending(
            category=row.category,
            total=_money(row.total),
            transaction_count=row.n,
            share_pct=(
                (_money(row.total) / total_spent * 100).quantize(_TENTH, ROUND_HALF_UP)
                if total_spent
                else Decimal("0.0")
            ),
        )
        for row in category_rows
    ]
    return SpendingReport(
        period=date_range,
        currency=currency,
        category_filter=category,
        total_spent=total_spent,
        transaction_count=sum(row.n for row in category_rows),
        by_category=by_category,
        top_merchants=[
            MerchantSpending(merchant=row.merchant, total=_money(row.total), transaction_count=row.n)
            for row in merchant_rows
        ],
    )
