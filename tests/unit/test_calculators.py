"""Exchange rates and the loan payment calculator."""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BankingProduct
from app.models.enums import Currency, ProductType
from app.services.errors import BusinessRuleError, NotFoundError
from app.services.exchange_rates import RATES_SOURCE, get_exchange_rate
from app.services.loans import annuity_payment, calculate_loan_payment

# --- annuity formula ----------------------------------------------------------


@pytest.mark.parametrize(
    ("principal", "rate", "months", "expected"),
    [
        # Textbook reference values, independently verifiable with any loan calculator.
        (Decimal("10000"), Decimal("12"), 12, Decimal("888.49")),
        (Decimal("200000"), Decimal("6"), 360, Decimal("1199.10")),
        (Decimal("1200"), Decimal("0"), 12, Decimal("100.00")),  # zero interest
        (Decimal("1000"), Decimal("24"), 1, Decimal("1020.00")),  # one month: P * (1 + r)
    ],
)
def test_annuity_payment_reference_values(
    principal: Decimal, rate: Decimal, months: int, expected: Decimal
) -> None:
    assert annuity_payment(principal, rate, months) == expected


@pytest.mark.parametrize(("principal", "months"), [(Decimal("1000"), 0), (Decimal("0"), 12)])
def test_annuity_payment_rejects_invalid_input(principal: Decimal, months: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        annuity_payment(principal, Decimal("10"), months)


# --- calculate_loan_payment (rates from the product table) ---------------------


async def test_rate_comes_from_the_product_catalogue(seeded_session: AsyncSession) -> None:
    quote = await calculate_loan_payment(
        seeded_session, product_code="LOAN_CONSUMER", amount=Decimal("10000"), term_months=24
    )

    assert quote.annual_interest_rate_pct == Decimal("14.50")
    assert quote.monthly_payment == annuity_payment(Decimal("10000"), Decimal("14.50"), 24)
    assert quote.total_payment == quote.monthly_payment * 24
    assert quote.total_interest == quote.total_payment - Decimal("10000")
    assert "MOCK" in quote.rate_source


@pytest.mark.parametrize(
    ("code", "amount", "months", "message"),
    [
        ("LOAN_CONSUMER", Decimal("100"), 24, "allows 500.00-50000.00 GEL"),
        ("LOAN_CONSUMER", Decimal("60000"), 24, "allows 500.00-50000.00 GEL"),
        ("LOAN_CONSUMER", Decimal("5000"), 72, "terms range from 6 to 60 months"),
        ("MORTGAGE_STANDARD", Decimal("100000"), 12, "terms range from 60 to 360 months"),
    ],
)
async def test_product_limits_are_enforced(
    seeded_session: AsyncSession, code: str, amount: Decimal, months: int, message: str
) -> None:
    with pytest.raises(BusinessRuleError, match=message):
        await calculate_loan_payment(seeded_session, product_code=code, amount=amount, term_months=months)


@pytest.mark.parametrize("code", ["LOAN_PAYDAY", "CARD_DEBIT_GOLD"])
async def test_unknown_or_non_loan_product_lists_alternatives(
    seeded_session: AsyncSession, code: str
) -> None:
    with pytest.raises(NotFoundError, match="Available: LOAN_AUTO, LOAN_CONSUMER, MORTGAGE_STANDARD"):
        await calculate_loan_payment(
            seeded_session, product_code=code, amount=Decimal("1000"), term_months=12
        )


# --- exchange rates -----------------------------------------------------------


def test_rate_and_conversion_are_reproducible_by_hand() -> None:
    quote = get_exchange_rate(Currency.USD, Currency.GEL, Decimal("100"))

    assert quote.rate == Decimal("2.7150")
    assert quote.converted_amount == Decimal("271.50")  # 100 x 2.7150
    assert quote.source == RATES_SOURCE
    assert "MOCK" in quote.source


def test_inverse_and_cross_rates() -> None:
    assert get_exchange_rate(Currency.GEL, Currency.USD).rate == Decimal("0.3683")  # 1 / 2.7150
    assert get_exchange_rate(Currency.USD, Currency.EUR).rate == Decimal("0.8532")  # 2.7150 / 3.1820


def test_rate_without_amount() -> None:
    quote = get_exchange_rate(Currency.EUR, Currency.GEL)
    assert (quote.amount, quote.converted_amount) == (None, None)


def test_same_currency_is_rejected() -> None:
    with pytest.raises(BusinessRuleError, match="differ"):
        get_exchange_rate(Currency.GEL, Currency.GEL)


async def test_loan_product_without_published_rate_is_refused(seeded_session: AsyncSession) -> None:
    seeded_session.add(
        BankingProduct(
            code="LOAN_DRAFT",
            name="Draft Loan",
            product_type=ProductType.LOAN,
            description="MOCK",
            is_active=True,
        )
    )
    await seeded_session.commit()

    with pytest.raises(BusinessRuleError, match="no published interest rate"):
        await calculate_loan_payment(
            seeded_session, product_code="LOAN_DRAFT", amount=Decimal("1000"), term_months=12
        )
