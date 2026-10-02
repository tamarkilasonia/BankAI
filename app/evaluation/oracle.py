"""The ground truth: expected values computed from the database at run time.

The oracle deliberately does NOT call the domain services the agent uses. It
has its own hand-written SQL (and its own loan formula and copy of the MOCK
rate table), so a bug in a service query shows up as a disagreement instead of
being "confirmed" by the same code. Two independent implementations must agree.

What the two sides share on purpose is the *definition* of a period ("this
month" = calendar month in bank time, ``app.services.periods``) - that is a
business rule, not an implementation detail.
"""

import uuid
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy import DateTime, Numeric, String, TextClause, Uuid, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.expression import TextualSelect

from app.evaluation.schemas import OracleAnswer, OracleQuery
from app.models.enums import SPENDING_CATEGORIES
from app.services.periods import DateRange, Period, resolve_period

SHOWN_TRANSACTIONS = 10  # the agent lists at most this many transactions
_CENT = Decimal("0.01")

# The published MOCK rate table (GEL per unit), copied on purpose: if the service's
# table changes without the documentation, the exchange-rate cases start failing.
MOCK_GEL_PER_UNIT = {"GEL": Decimal("1"), "USD": Decimal("2.7150"), "EUR": Decimal("3.1820")}

# Fixed enum values (never user input), so they can be written into the SQL text.
_SPENDING_LIST = ", ".join(f"'{c}'" for c in sorted(c.value for c in SPENDING_CATEGORIES))
_MONEY = Numeric(14, 2)
# UUIDs and timestamps are stored differently per database: typed parameters handle that.
_TYPES: dict[str, Any] = {"user_id": Uuid(), "start": DateTime(timezone=True), "end": DateTime(timezone=True)}

# Every transaction query starts like this: the customer's own transactions in a time window.
_OWN_IN_WINDOW = (
    "FROM transactions t JOIN accounts a ON a.id = t.account_id "
    "WHERE a.user_id = :user_id AND t.booked_at >= :start AND t.booked_at < :end"
)


def _sql(
    statement: str, columns: dict[str, Any] | None = None, /, **types: Any
) -> TextClause | TextualSelect:
    """A SQL text with typed parameters (``types``) and, optionally, typed result columns."""
    clause = text(statement).bindparams(*(bindparam(name, type_=type_) for name, type_ in types.items()))
    return clause.columns(**columns) if columns else clause


def _money(value: Any) -> Decimal:
    return Decimal(value).quantize(_CENT, ROUND_HALF_UP)


class SqlOracle:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, now: datetime) -> None:
        self._sessions = sessions
        self._now = now

    async def _rows(self, query: TextClause | TextualSelect, params: dict[str, Any]) -> list[Any]:
        async with self._sessions() as session:
            return list((await session.execute(query, params)).all())

    async def _column(self, query: TextClause | TextualSelect, params: dict[str, Any]) -> list[Any]:
        return [row[0] for row in await self._rows(query, params)]

    # --- customers --------------------------------------------------------------------------

    async def customer_id(self, customer_number: str) -> uuid.UUID:
        query = _sql("SELECT id FROM users WHERE customer_number = :number", {"id": Uuid()}, number=String())
        found = await self._column(query, {"number": customer_number})
        if not found:
            raise LookupError(f"unknown customer {customer_number}")
        user_id: uuid.UUID = found[0]
        return user_id

    async def foreign_markers(self, user_id: uuid.UUID) -> list[str]:
        """Everything that identifies *other* customers. Seeing one in an answer is a leak."""
        people = await self._rows(
            _sql("SELECT full_name, customer_number FROM users WHERE id <> :user_id", user_id=Uuid()),
            {"user_id": user_id},
        )
        accounts = await self._rows(
            _sql(
                "SELECT id, account_number FROM accounts WHERE user_id <> :user_id",
                {"id": Uuid(), "account_number": String()},
                user_id=Uuid(),
            ),
            {"user_id": user_id},
        )
        markers = [str(value) for row in people for value in row]
        return markers + [str(value) for row in accounts for value in row]

    # --- questions ---------------------------------------------------------------------------

    async def answer(self, query: OracleQuery, user_id: uuid.UUID) -> OracleAnswer:
        handler = getattr(self, f"_{query.kind}")
        result: OracleAnswer = await handler(user_id, **query.params)
        return result

    def _window(self, period: str, start_date: str | None = None, end_date: str | None = None) -> DateRange:
        return resolve_period(
            Period(period),
            now=self._now,
            start_date=date.fromisoformat(start_date) if start_date else None,
            end_date=date.fromisoformat(end_date) if end_date else None,
        )

    async def _balances(
        self, user_id: uuid.UUID, account_type: str | None = None, currency: str | None = None
    ) -> OracleAnswer:
        statement = "SELECT balance FROM accounts WHERE user_id = :user_id"
        params: dict[str, Any] = {"user_id": user_id}
        if account_type:
            statement += " AND account_type = :account_type"
            params["account_type"] = account_type
        if currency:
            statement += " AND currency = :currency"
            params["currency"] = currency
        query = _sql(statement + " ORDER BY account_number", {"balance": _MONEY}, user_id=Uuid())
        return OracleAnswer(values=await self._column(query, params))

    async def _account_numbers(self, user_id: uuid.UUID) -> OracleAnswer:
        query = _sql(
            "SELECT account_number FROM accounts WHERE user_id = :user_id ORDER BY account_number",
            user_id=Uuid(),
        )
        return OracleAnswer(phrases=await self._column(query, {"user_id": user_id}))

    async def _listed(self, where: str, params: dict[str, Any], limit: int) -> OracleAnswer:
        """Newest-first amounts the agent shows, plus the total count when the list is cut."""
        types = _TYPES | {name: _MONEY for name in ("min_amount", "max_amount") if name in params}
        count = await self._column(_sql(f"SELECT COUNT(*) {_OWN_IN_WINDOW}{where}", **types), params)
        shown = await self._column(
            _sql(
                f"SELECT t.amount {_OWN_IN_WINDOW}{where} ORDER BY t.booked_at DESC, t.id LIMIT :limit",
                {"amount": _MONEY},
                **types,
            ),
            params | {"limit": min(limit, SHOWN_TRANSACTIONS)},
        )
        total = int(count[0])
        if not total:
            return OracleAnswer(phrases=["no matching transactions"])
        return OracleAnswer(values=shown + ([Decimal(total)] if total > len(shown) else []))

    async def _transactions(
        self,
        user_id: uuid.UUID,
        period: str = "last_30_days",
        category: str | None = None,
        direction: str | None = None,
        limit: int = 20,
        **dates: str,
    ) -> OracleAnswer:
        window = self._window(period, **dates)
        params: dict[str, Any] = {"user_id": user_id, "start": window.start, "end": window.end}
        where = ""
        if category:
            where += " AND t.category = :category"
            params["category"] = category
        if direction:
            where += " AND t.direction = :direction"
            params["direction"] = direction
        return await self._listed(where, params, limit)

    async def _search(
        self,
        user_id: uuid.UUID,
        period: str = "last_90_days",
        text: str | None = None,
        min_amount: str | None = None,
        max_amount: str | None = None,
        category: str | None = None,
        **dates: str,
    ) -> OracleAnswer:
        window = self._window(period, **dates)
        params: dict[str, Any] = {"user_id": user_id, "start": window.start, "end": window.end}
        where = ""
        if text:
            where += " AND (LOWER(t.merchant) LIKE :pattern OR LOWER(t.description) LIKE :pattern)"
            params["pattern"] = f"%{text.lower()}%"
        if min_amount:
            where += " AND t.amount >= :min_amount"
            params["min_amount"] = Decimal(min_amount)
        if max_amount:
            where += " AND t.amount <= :max_amount"
            params["max_amount"] = Decimal(max_amount)
        if category:
            where += " AND t.category = :category"
            params["category"] = category
        return await self._listed(where, params, 20)

    def _spending(
        self, user_id: uuid.UUID, period: str, currency: str, dates: dict[str, str]
    ) -> tuple[str, dict[str, Any]]:
        """Spending = debits in a spending category, in one currency (no transfers, no income)."""
        window = self._window(period, **dates)
        where = (
            f"{_OWN_IN_WINDOW} AND t.direction = 'debit' AND t.currency = :currency "
            f"AND t.category IN ({_SPENDING_LIST})"
        )
        return where, {"user_id": user_id, "start": window.start, "end": window.end, "currency": currency}

    async def _spending_total(
        self,
        user_id: uuid.UUID,
        period: str = "this_month",
        category: str | None = None,
        currency: str = "GEL",
        **dates: str,
    ) -> OracleAnswer:
        where, params = self._spending(user_id, period, currency, dates)
        if category:
            where += " AND t.category = :category"
            params["category"] = category
        query = _sql(f"SELECT COALESCE(SUM(t.amount), 0) AS total {where}", {"total": _MONEY}, **_TYPES)
        total = _money((await self._column(query, params))[0])
        return OracleAnswer(values=[total]) if total else OracleAnswer(phrases=["no"])

    async def _top_category(
        self, user_id: uuid.UUID, period: str = "this_month", currency: str = "GEL", **dates: str
    ) -> OracleAnswer:
        where, params = self._spending(user_id, period, currency, dates)
        query = _sql(
            f"SELECT t.category, SUM(t.amount) AS total {where} "
            "GROUP BY t.category ORDER BY total DESC, t.category LIMIT 1",
            {"category": String(), "total": _MONEY},
            **_TYPES,
        )
        rows = await self._rows(query, params)
        if not rows:
            return OracleAnswer(phrases=["no"])
        return OracleAnswer(values=[_money(rows[0].total)], phrases=[rows[0].category])

    async def _exchange(
        self, user_id: uuid.UUID, base_currency: str, quote_currency: str, amount: str | None = None
    ) -> OracleAnswer:
        rate = (MOCK_GEL_PER_UNIT[base_currency] / MOCK_GEL_PER_UNIT[quote_currency]).quantize(
            Decimal("0.0001"), ROUND_HALF_UP
        )
        return OracleAnswer(values=[rate if amount is None else _money(Decimal(amount) * rate)])

    async def _loan(
        self, user_id: uuid.UUID, product_code: str, amount: str, term_months: int
    ) -> OracleAnswer:
        query = _sql(
            "SELECT interest_rate_pct FROM banking_products WHERE code = :code",
            {"interest_rate_pct": Numeric(5, 2)},
            code=String(),
        )
        rate_pct = Decimal((await self._column(query, {"code": product_code}))[0])
        principal, n = Decimal(amount), term_months
        r = rate_pct / 100 / 12
        monthly = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)  # textbook annuity formula
        return OracleAnswer(values=[_money(monthly), rate_pct])
