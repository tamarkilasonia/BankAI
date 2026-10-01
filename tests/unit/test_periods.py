"""Period resolution - calendar logic in bank time (UTC+4)."""

from datetime import UTC, date, datetime

import pytest

from app.services.errors import BusinessRuleError
from app.services.periods import MAX_RANGE_DAYS, Period, resolve_period
from tests.constants import ANCHOR  # Monday 2026-09-28, 22:00 bank time


@pytest.mark.parametrize(
    ("period", "start", "end"),
    [
        (Period.TODAY, date(2026, 9, 28), date(2026, 9, 28)),
        (Period.THIS_MONTH, date(2026, 9, 1), date(2026, 9, 28)),
        (Period.LAST_MONTH, date(2026, 8, 1), date(2026, 8, 31)),
        (Period.LAST_7_DAYS, date(2026, 9, 22), date(2026, 9, 28)),
        (Period.LAST_30_DAYS, date(2026, 8, 30), date(2026, 9, 28)),
        (Period.LAST_90_DAYS, date(2026, 7, 1), date(2026, 9, 28)),
        (Period.THIS_YEAR, date(2026, 1, 1), date(2026, 9, 28)),
    ],
)
def test_named_periods(period: Period, start: date, end: date) -> None:
    resolved = resolve_period(period, now=ANCHOR)

    assert (resolved.start_date, resolved.end_date) == (start, end)
    assert (resolved.end_date - resolved.start_date).days + 1 <= MAX_RANGE_DAYS


def test_month_is_decided_in_bank_time_not_utc() -> None:
    # 21:30 UTC on 30 Sep is already 01:30 on 1 Oct in Tbilisi.
    late_evening_utc = datetime(2026, 9, 30, 21, 30, tzinfo=UTC)

    resolved = resolve_period(Period.THIS_MONTH, now=late_evening_utc)

    assert resolved.start_date == date(2026, 10, 1)


def test_last_month_in_january_is_previous_december() -> None:
    resolved = resolve_period(Period.LAST_MONTH, now=datetime(2027, 1, 15, 12, tzinfo=UTC))

    assert (resolved.start_date, resolved.end_date) == (date(2026, 12, 1), date(2026, 12, 31))


def test_utc_bounds_are_half_open_bank_midnights() -> None:
    resolved = resolve_period(Period.THIS_MONTH, now=ANCHOR)

    # Midnight in Tbilisi is 20:00 UTC the evening before.
    assert resolved.start == datetime(2026, 8, 31, 20, 0, tzinfo=UTC)
    assert resolved.end == datetime(2026, 9, 28, 20, 0, tzinfo=UTC)  # exclusive


def test_custom_period() -> None:
    resolved = resolve_period(
        Period.CUSTOM, now=ANCHOR, start_date=date(2026, 7, 10), end_date=date(2026, 7, 20)
    )

    assert (resolved.start_date, resolved.end_date) == (date(2026, 7, 10), date(2026, 7, 20))


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (None, date(2026, 7, 1), "both start_date and end_date"),
        (date(2026, 7, 1), None, "both start_date and end_date"),
        (date(2026, 7, 2), date(2026, 7, 1), "must not be before"),
        (date(2025, 1, 1), date(2026, 9, 1), "at most"),
    ],
)
def test_invalid_custom_periods(start: date | None, end: date | None, message: str) -> None:
    with pytest.raises(BusinessRuleError, match=message):
        resolve_period(Period.CUSTOM, now=ANCHOR, start_date=start, end_date=end)
