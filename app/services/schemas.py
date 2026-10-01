"""Result models returned by the services (and, unchanged, by the MCP tools).

Money fields are ``Decimal``; in JSON they serialize as strings ("12.30") so no
precision is lost between the database, the MCP server and the agent.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.models.enums import AccountStatus, AccountType, Currency, TransactionCategory, TransactionDirection
from app.services.periods import DateRange

MOCK_DATA_NOTICE = "MOCK data - BankAI is a prototype and holds no real customer information."


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- accounts -------------------------------------------------------------------


class AccountSummary(_Result):
    account_id: uuid.UUID
    account_number: str
    name: str
    account_type: AccountType
    currency: Currency
    balance: Decimal
    status: AccountStatus


class BalanceResult(_Result):
    accounts: list[AccountSummary]
    # Balances in different currencies are never added together.
    totals_by_currency: dict[Currency, Decimal]
    as_of: datetime
    notice: str = MOCK_DATA_NOTICE


class AccountDetails(AccountSummary):
    opened_at: datetime


class AccountDetailsResult(_Result):
    customer_number: str
    full_name: str
    accounts: list[AccountDetails]
    notice: str = MOCK_DATA_NOTICE


# --- transactions ---------------------------------------------------------------


class TransactionItem(_Result):
    transaction_id: uuid.UUID
    account_id: uuid.UUID
    booked_at: datetime
    direction: TransactionDirection
    amount: Decimal
    currency: Currency
    category: TransactionCategory
    merchant: str
    description: str


class TransactionList(_Result):
    period: DateRange
    transactions: list[TransactionItem]
    total_matches: int  # how many matched in total; `transactions` may be limited
    truncated: bool
    notice: str = MOCK_DATA_NOTICE


# --- spending -------------------------------------------------------------------


class CategorySpending(_Result):
    category: TransactionCategory
    total: Decimal
    transaction_count: int
    share_pct: Decimal  # share of total spending in the period, in percent


class MerchantSpending(_Result):
    merchant: str
    total: Decimal
    transaction_count: int


class SpendingReport(_Result):
    period: DateRange
    currency: Currency
    category_filter: TransactionCategory | None
    total_spent: Decimal
    transaction_count: int
    by_category: list[CategorySpending]
    top_merchants: list[MerchantSpending]
    notice: str = MOCK_DATA_NOTICE


# --- exchange rates & calculators ----------------------------------------------


class ExchangeRateQuote(_Result):
    base_currency: Currency
    quote_currency: Currency
    rate: Decimal  # 1 unit of base = `rate` units of quote
    amount: Decimal | None
    converted_amount: Decimal | None
    as_of: date
    source: str


class LoanPaymentQuote(_Result):
    product_code: str
    product_name: str
    currency: Currency
    principal: Decimal
    term_months: int
    annual_interest_rate_pct: Decimal
    monthly_payment: Decimal
    total_payment: Decimal
    total_interest: Decimal
    rate_source: str
    note: str
