"""Properties of the money and date calculations (Hypothesis).

Example tests check a handful of inputs; these check rules that must hold for all of
them - over decades of dates, every loan size and rate, every currency pair.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from app.models.enums import Currency
from app.services.errors import BusinessRuleError
from app.services.exchange_rates import get_exchange_rate
from app.services.loans import annuity_payment
from app.services.periods import MAX_RANGE_DAYS, Period, resolve_period

PRINCIPALS = st.decimals(min_value=Decimal("100"), max_value=Decimal("10000000"), places=2)
RATES = st.decimals(min_value=Decimal("0"), max_value=Decimal("40"), places=2)
TERMS = st.integers(1, 480)
MOMENTS = st.datetimes(
    # Hypothesis wants naive bounds and adds the time zone itself.
    min_value=datetime(2000, 1, 1),  # noqa: DTZ001
    max_value=datetime(2099, 12, 31),  # noqa: DTZ001
    timezones=st.just(UTC),
)
CURRENCY_PAIRS = st.permutations(list(Currency)).map(lambda c: (c[0], c[1]))


# --- loans ---------------------------------------------------------------------------------------


@given(PRINCIPALS, RATES, TERMS)
def test_a_loan_is_always_repaid_in_full(principal: Decimal, rate: Decimal, term: int) -> None:
    payment = annuity_payment(principal, rate, term)
    assert payment > 0
    # n payments repay the principal (up to half a cent of rounding per payment) ...
    assert payment * term >= principal - Decimal("0.005") * term
    # ... and never more than simple interest on the whole principal for the whole term would.
    assert payment * term <= principal * (1 + rate / 100 * term / 12) + Decimal("0.005") * term


@given(PRINCIPALS, RATES, RATES, TERMS)
def test_a_higher_rate_never_lowers_the_payment(
    principal: Decimal, r1: Decimal, r2: Decimal, term: int
) -> None:
    low, high = sorted([r1, r2])
    assert annuity_payment(principal, low, term) <= annuity_payment(principal, high, term)


@given(PRINCIPALS, TERMS, TERMS)
def test_a_longer_term_never_raises_the_monthly_payment(principal: Decimal, t1: int, t2: int) -> None:
    short, long = sorted([t1, t2])
    assert annuity_payment(principal, Decimal("14.5"), long) <= annuity_payment(
        principal, Decimal("14.5"), short
    )


# --- exchange rates ------------------------------------------------------------------------------


@given(CURRENCY_PAIRS, PRINCIPALS)
def test_converting_there_and_back_loses_at_most_rounding(
    pair: tuple[Currency, Currency], amount: Decimal
) -> None:
    there = get_exchange_rate(pair[0], pair[1], amount)
    assert there.converted_amount is not None and there.converted_amount > 0
    back = get_exchange_rate(pair[1], pair[0], there.converted_amount)
    assert back.converted_amount is not None
    # Each displayed rate is off by at most half a unit in its 4th decimal (relative: 0.00005 / rate),
    # and each conversion by half a cent - the first one then multiplied by the second rate.
    rate_error = Decimal("0.00005") * (1 / there.rate + 1 / back.rate) * Decimal("1.01")
    cents = Decimal("0.005") * (back.rate + 1)
    assert abs(back.converted_amount - amount) <= amount * rate_error + cents
    assert "MOCK" in there.source


# --- periods -------------------------------------------------------------------------------------

ROLLING = {Period.TODAY: 1, Period.LAST_7_DAYS: 7, Period.LAST_30_DAYS: 30, Period.LAST_90_DAYS: 90}


@given(MOMENTS)
def test_every_named_period_is_a_sane_range_ending_today(now: datetime) -> None:
    ranges = {p: resolve_period(p, now=now) for p in Period if p is not Period.CUSTOM}
    today = ranges[Period.TODAY].start_date

    for period, r in ranges.items():
        assert r.start_date <= r.end_date and (r.end_date - r.start_date).days < MAX_RANGE_DAYS
        assert r.end - r.start == timedelta(days=(r.end_date - r.start_date).days + 1)  # whole days, no DST
        if period is not Period.LAST_MONTH:
            assert r.end_date == today
    for period, days in ROLLING.items():
        assert (ranges[period].end_date - ranges[period].start_date).days + 1 == days
    this_month, last_month = ranges[Period.THIS_MONTH], ranges[Period.LAST_MONTH]
    assert this_month.start_date.day == 1 and last_month.start_date.day == 1
    assert last_month.end_date + timedelta(days=1) == this_month.start_date  # adjacent, no gap, no overlap
    assert ranges[Period.THIS_YEAR].start_date == date(today.year, 1, 1)


@given(MOMENTS, st.dates(date(2000, 1, 1), date(2099, 12, 31)), st.integers(-5, 400))
def test_a_custom_range_is_accepted_exactly_when_it_is_valid(now: datetime, start: date, length: int) -> None:
    end = start + timedelta(days=length - 1)
    assume(end.year < 2100)
    if 1 <= length <= MAX_RANGE_DAYS:
        r = resolve_period(Period.CUSTOM, now=now, start_date=start, end_date=end)
        assert (r.start_date, r.end_date) == (start, end)
    else:
        with pytest.raises(BusinessRuleError):
            resolve_period(Period.CUSTOM, now=now, start_date=start, end_date=end)
