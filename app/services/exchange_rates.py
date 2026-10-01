"""Currency conversion with a MOCK, static rate table.

These are illustrative numbers, NOT real market rates, and every quote says
so in its ``source`` field. A production system would read rates from a
market-data feed with timestamps and buy/sell spreads.
"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from app.models.enums import Currency
from app.services.errors import BusinessRuleError
from app.services.schemas import ExchangeRateQuote

RATES_AS_OF = date(2026, 9, 1)
RATES_SOURCE = "MOCK static rate table - illustrative values, not real market rates"

# How many GEL one unit of each currency is worth (MOCK mid-market values).
_GEL_PER_UNIT: dict[Currency, Decimal] = {
    Currency.GEL: Decimal("1"),
    Currency.USD: Decimal("2.7150"),
    Currency.EUR: Decimal("3.1820"),
}

_RATE_PRECISION = Decimal("0.0001")
_CENT = Decimal("0.01")


def get_exchange_rate(
    base_currency: Currency, quote_currency: Currency, amount: Decimal | None = None
) -> ExchangeRateQuote:
    """Rate for 1 unit of ``base_currency`` in ``quote_currency``, optionally converting an amount.

    Cross rates (USD -> EUR) go through GEL. The conversion uses the *displayed*
    4-decimal rate, so a customer can reproduce the result by hand.
    """
    if base_currency == quote_currency:
        raise BusinessRuleError("Base and quote currency must differ.")
    rate = (_GEL_PER_UNIT[base_currency] / _GEL_PER_UNIT[quote_currency]).quantize(
        _RATE_PRECISION, ROUND_HALF_UP
    )
    converted = (amount * rate).quantize(_CENT, ROUND_HALF_UP) if amount is not None else None
    return ExchangeRateQuote(
        base_currency=base_currency,
        quote_currency=quote_currency,
        rate=rate,
        amount=amount,
        converted_amount=converted,
        as_of=RATES_AS_OF,
        source=RATES_SOURCE,
    )
