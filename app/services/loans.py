"""Loan payment calculator.

The interest rate is never supplied by the caller (or the LLM): it is read from
the ``banking_products`` table, and the request must respect the product's
amount and term limits. This is one of the anti-hallucination guarantees -
the assistant cannot quote a made-up rate.
"""

from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BankingProduct
from app.models.enums import ProductType
from app.services.errors import BusinessRuleError, NotFoundError
from app.services.schemas import LoanPaymentQuote

LOAN_PRODUCT_TYPES = frozenset({ProductType.LOAN, ProductType.MORTGAGE})
_CENT = Decimal("0.01")


def annuity_payment(principal: Decimal, annual_rate_pct: Decimal, term_months: int) -> Decimal:
    """Fixed monthly payment of an amortizing (annuity) loan.

        payment = P * r / (1 - (1 + r) ** -n)

    where P is the principal, r the monthly rate (annual % / 12 / 100) and n
    the number of months. With r = 0 the payment is simply P / n.
    """
    if term_months <= 0:
        raise ValueError("term_months must be positive")
    if principal <= 0:
        raise ValueError("principal must be positive")
    monthly_rate = annual_rate_pct / Decimal(1200)
    if monthly_rate == 0:
        payment = principal / term_months
    else:
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
    return payment.quantize(_CENT, ROUND_HALF_UP)


async def calculate_loan_payment(
    session: AsyncSession, *, product_code: str, amount: Decimal, term_months: int
) -> LoanPaymentQuote:
    product = await session.scalar(
        select(BankingProduct).where(BankingProduct.code == product_code, BankingProduct.is_active.is_(True))
    )
    if product is None or product.product_type not in LOAN_PRODUCT_TYPES:
        codes = await session.scalars(
            select(BankingProduct.code)
            .where(BankingProduct.product_type.in_(LOAN_PRODUCT_TYPES), BankingProduct.is_active.is_(True))
            .order_by(BankingProduct.code)
        )
        raise NotFoundError(f"Unknown loan product '{product_code}'. Available: {', '.join(codes)}.")
    if product.interest_rate_pct is None:
        raise BusinessRuleError(f"{product.name} has no published interest rate.")

    limits = f"{product.name} allows {product.min_amount}-{product.max_amount} {product.currency.value}"
    if (product.min_amount is not None and amount < product.min_amount) or (
        product.max_amount is not None and amount > product.max_amount
    ):
        raise BusinessRuleError(f"{limits}; requested {amount}.")
    if (product.min_term_months is not None and term_months < product.min_term_months) or (
        product.max_term_months is not None and term_months > product.max_term_months
    ):
        raise BusinessRuleError(
            f"{product.name} terms range from {product.min_term_months} to {product.max_term_months} months;"
            f" requested {term_months}."
        )

    monthly = annuity_payment(amount, product.interest_rate_pct, term_months)
    total = (monthly * term_months).quantize(_CENT)
    return LoanPaymentQuote(
        product_code=product.code,
        product_name=product.name,
        currency=product.currency,
        principal=amount,
        term_months=term_months,
        annual_interest_rate_pct=product.interest_rate_pct,
        monthly_payment=monthly,
        total_payment=total,
        total_interest=(total - amount).quantize(_CENT),
        rate_source="banking_products table (MOCK product catalogue)",
        note="Estimate for a fixed-rate annuity loan; excludes fees and insurance. Not a credit offer.",
    )
