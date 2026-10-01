"""Turning "this month" / "last 30 days" into exact date ranges.

Ranges are whole calendar days in bank time (UTC+4), inclusive of both
``start_date`` and ``end_date``. For SQL they become a half-open UTC interval
``[start, end)``: ``start`` = midnight of the first day, ``end`` = midnight
after the last day. Half-open intervals never double-count a transaction that
sits exactly on a boundary.
"""

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.core.time import BANK_TZ
from app.services.errors import BusinessRuleError

MAX_RANGE_DAYS = 366


class Period(StrEnum):
    TODAY = "today"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"


_ROLLING_DAYS = {Period.LAST_7_DAYS: 7, Period.LAST_30_DAYS: 30, Period.LAST_90_DAYS: 90}


class DateRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    period: Period
    start_date: date
    end_date: date

    @property
    def start(self) -> datetime:
        """Inclusive lower bound (UTC)."""
        return datetime.combine(self.start_date, time(0), tzinfo=BANK_TZ).astimezone(UTC)

    @property
    def end(self) -> datetime:
        """Exclusive upper bound (UTC): midnight after ``end_date``."""
        return datetime.combine(self.end_date + timedelta(days=1), time(0), tzinfo=BANK_TZ).astimezone(UTC)


def resolve_period(
    period: Period,
    *,
    now: datetime,
    start_date: date | None = None,
    end_date: date | None = None,
) -> DateRange:
    today = now.astimezone(BANK_TZ).date()

    if period is Period.CUSTOM:
        if start_date is None or end_date is None:
            raise BusinessRuleError("A custom period needs both start_date and end_date.")
        if end_date < start_date:
            raise BusinessRuleError("end_date must not be before start_date.")
        if (end_date - start_date).days + 1 > MAX_RANGE_DAYS:
            raise BusinessRuleError(f"A period can cover at most {MAX_RANGE_DAYS} days.")
        return DateRange(period=period, start_date=start_date, end_date=end_date)

    if period is Period.TODAY:
        first = today
    elif period is Period.THIS_MONTH:
        first = today.replace(day=1)
    elif period is Period.LAST_MONTH:
        last_of_previous = today.replace(day=1) - timedelta(days=1)
        return DateRange(period=period, start_date=last_of_previous.replace(day=1), end_date=last_of_previous)
    elif period is Period.THIS_YEAR:
        first = today.replace(month=1, day=1)
    else:
        first = today - timedelta(days=_ROLLING_DAYS[period] - 1)
    return DateRange(period=period, start_date=first, end_date=today)
