"""The tool contract shared by the MCP server and the agent's MCP client.

Each tool has exactly one input model and one output model:

* The **server** publishes its MCP JSON schema from the input model and
  validates every call against it (including cross-field rules).
* The **client gateway** validates arguments with the same model *before*
  sending them - so arguments invented by an LLM are checked twice.

Input models use ``extra="forbid"``: an argument the tool does not declare
(for example a sneaked-in ``user_id``) is rejected instead of being ignored.
Note what is *not* in any input model: the customer's identity. It travels
separately as a signed token (see ``app.core.security``).

"""

import uuid
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.models.enums import (
    SPENDING_CATEGORIES,
    AccountType,
    Currency,
    TransactionCategory,
    TransactionDirection,
)
from app.rag.schemas import PolicySearchResult
from app.services.periods import Period
from app.services.schemas import (
    AccountDetailsResult,
    BalanceResult,
    ExchangeRateQuote,
    LoanPaymentQuote,
    SpendingReport,
    TransactionList,
)

_ACCOUNT_ID = "Restrict to one account (account_id from get_account_details). Omit for all accounts."
_MAX_MONEY = Decimal("10000000")


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PeriodInput(ToolInput):
    """Shared period arguments. Dates are bank-calendar days, inclusive."""

    period: Period = Field(default=Period.THIS_MONTH, description="Time window to look at.")
    start_date: date | None = Field(default=None, description="Only with period='custom' (YYYY-MM-DD).")
    end_date: date | None = Field(default=None, description="Only with period='custom' (YYYY-MM-DD).")

    @model_validator(mode="after")
    def _custom_dates(self) -> Self:
        has_dates = self.start_date is not None or self.end_date is not None
        if self.period is Period.CUSTOM:
            if self.start_date is None or self.end_date is None:
                raise ValueError("period 'custom' requires both start_date and end_date")
            if self.end_date < self.start_date:
                raise ValueError("end_date must not be before start_date")
        elif has_dates:
            raise ValueError("start_date/end_date are only allowed with period 'custom'")
        return self


class GetAccountBalanceInput(ToolInput):
    account_id: uuid.UUID | None = Field(default=None, description=_ACCOUNT_ID)
    account_type: AccountType | None = Field(default=None, description="Only accounts of this type.")
    currency: Currency | None = Field(default=None, description="Only accounts in this currency.")


class GetAccountDetailsInput(ToolInput):
    account_id: uuid.UUID | None = Field(default=None, description=_ACCOUNT_ID)


class GetTransactionsInput(PeriodInput):
    period: Period = Field(default=Period.LAST_30_DAYS, description="Time window to look at.")
    account_id: uuid.UUID | None = Field(default=None, description=_ACCOUNT_ID)
    category: TransactionCategory | None = Field(default=None, description="Only this category.")
    direction: TransactionDirection | None = Field(
        default=None, description="'debit' = money out, 'credit' = money in."
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum transactions to return (newest first).")


class SearchTransactionsInput(PeriodInput):
    period: Period = Field(default=Period.LAST_90_DAYS, description="Time window to search.")
    text: str | None = Field(
        default=None,
        min_length=2,
        max_length=80,
        description="Case-insensitive text in merchant or description.",
    )
    min_amount: Decimal | None = Field(default=None, ge=0, le=_MAX_MONEY, decimal_places=2)
    max_amount: Decimal | None = Field(default=None, ge=0, le=_MAX_MONEY, decimal_places=2)
    category: TransactionCategory | None = None
    account_id: uuid.UUID | None = Field(default=None, description=_ACCOUNT_ID)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _has_criterion(self) -> Self:
        if (
            self.text is None
            and self.min_amount is None
            and self.max_amount is None
            and self.category is None
        ):
            raise ValueError("give at least one of: text, min_amount, max_amount, category")
        if self.min_amount is not None and self.max_amount is not None and self.min_amount > self.max_amount:
            raise ValueError("min_amount must not be greater than max_amount")
        return self


class AnalyzeSpendingInput(PeriodInput):
    currency: Currency = Field(default=Currency.GEL, description="Spending in this currency only.")
    category: TransactionCategory | None = Field(default=None, description="Only this spending category.")
    account_id: uuid.UUID | None = Field(default=None, description=_ACCOUNT_ID)
    top_merchants: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def _spending_category(self) -> Self:
        if self.category is not None and self.category not in SPENDING_CATEGORIES:
            raise ValueError(f"'{self.category.value}' is not a spending category")
        return self


class GetExchangeRateInput(ToolInput):
    base_currency: Currency = Field(description="Currency to convert from.")
    quote_currency: Currency = Field(description="Currency to convert to.")
    amount: Decimal | None = Field(
        default=None, gt=0, le=_MAX_MONEY, decimal_places=2, description="Optional amount to convert."
    )

    @model_validator(mode="after")
    def _different_currencies(self) -> Self:
        if self.base_currency == self.quote_currency:
            raise ValueError("base_currency and quote_currency must differ")
        return self


class SearchBankingPolicyInput(ToolInput):
    query: str = Field(min_length=3, max_length=500, description="The customer's question about bank policy.")
    top_k: int = Field(default=4, ge=1, le=8, description="How many passages to return at most.")


class CalculateLoanPaymentInput(ToolInput):
    product_code: str = Field(
        pattern=r"^[A-Z][A-Z0-9_]{2,39}$",
        description="Loan product code, e.g. LOAN_CONSUMER, LOAN_AUTO or MORTGAGE_STANDARD.",
    )
    amount: Decimal = Field(gt=0, le=_MAX_MONEY, decimal_places=2, description="Amount to borrow.")
    term_months: int = Field(ge=1, le=480, description="Repayment period in months.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[ToolInput]
    output_model: type[BaseModel]


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "get_account_balance",
            "Account balances",
            "Current balances of the customer's own accounts, with totals per currency.",
            GetAccountBalanceInput,
            BalanceResult,
        ),
        ToolSpec(
            "get_account_details",
            "Account details",
            "The customer's profile basics and account list: IDs, numbers, types, currencies, status.",
            GetAccountDetailsInput,
            AccountDetailsResult,
        ),
        ToolSpec(
            "get_transactions",
            "Recent transactions",
            "The customer's transactions in a period, newest first, optionally by category or direction.",
            GetTransactionsInput,
            TransactionList,
        ),
        ToolSpec(
            "search_transactions",
            "Search transactions",
            "Find the customer's transactions by merchant/description text and/or amount range.",
            SearchTransactionsInput,
            TransactionList,
        ),
        ToolSpec(
            "analyze_spending",
            "Spending analysis",
            "Total spending in a period, broken down by category and top merchants (one currency).",
            AnalyzeSpendingInput,
            SpendingReport,
        ),
        ToolSpec(
            "get_exchange_rate",
            "Exchange rate",
            "MOCK exchange rate between GEL, USD and EUR, optionally converting an amount.",
            GetExchangeRateInput,
            ExchangeRateQuote,
        ),
        ToolSpec(
            "calculate_loan_payment",
            "Loan payment calculator",
            "Monthly payment for a loan product, using the product's published interest rate.",
            CalculateLoanPaymentInput,
            LoanPaymentQuote,
        ),
        ToolSpec(
            "search_banking_policy",
            "Bank policy search",
            "Search the bank's policy documents (fees, limits, cards, loans, security...).",
            SearchBankingPolicyInput,
            PolicySearchResult,
        ),
    )
}


def describe_validation_error(exc: ValidationError) -> str:
    """A compact, safe summary: field names and rules, never the rejected values."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
