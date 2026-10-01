"""Independent expected values for service tests.

Computed in plain Python straight from the generated mock data, without any
SQL - so a bug in a service's query cannot also hide in the expected value.
"""

import uuid
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from app.database.seed import SeedData, generate_seed_data
from app.models import Transaction
from app.models.enums import SPENDING_CATEGORIES, Currency, TransactionDirection
from app.services.periods import DateRange
from tests.constants import ANCHOR

SEED: SeedData = generate_seed_data(anchor=ANCHOR)


def user_transactions(user_id: uuid.UUID, start: datetime, end: datetime) -> list[Transaction]:
    owned = {a.id for a in SEED.accounts if a.user_id == user_id}
    return [t for t in SEED.transactions if t.account_id in owned and start <= t.booked_at < end]


def spending_by_category(
    user_id: uuid.UUID, date_range: DateRange, currency: Currency = Currency.GEL
) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for tx in user_transactions(user_id, date_range.start, date_range.end):
        if (
            tx.direction is TransactionDirection.DEBIT
            and tx.currency is currency
            and tx.category in SPENDING_CATEGORIES
        ):
            totals[tx.category.value] += tx.amount
    return dict(totals)
