"""Domain enumerations.

They are ``StrEnum`` so they serialize to plain strings in JSON and in the
database. In the database they are stored as VARCHAR (``native_enum=False``)
rather than PostgreSQL ENUM types, because adding a value to a native enum
requires a special migration; a VARCHAR column does not.
"""

from enum import StrEnum


class Language(StrEnum):
    ENGLISH = "en"
    GEORGIAN = "ka"


class Currency(StrEnum):
    GEL = "GEL"
    USD = "USD"
    EUR = "EUR"


class AccountType(StrEnum):
    CHECKING = "checking"
    SAVINGS = "savings"


class AccountStatus(StrEnum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


class TransactionDirection(StrEnum):
    DEBIT = "debit"  # money leaves the account
    CREDIT = "credit"  # money enters the account


class TransactionCategory(StrEnum):
    GROCERIES = "groceries"
    RESTAURANTS = "restaurants"
    TRANSPORTATION = "transportation"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    SUBSCRIPTIONS = "subscriptions"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    SALARY = "salary"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"


# Categories that represent spending (used by spending analysis).
SPENDING_CATEGORIES: frozenset[TransactionCategory] = frozenset(
    {
        TransactionCategory.GROCERIES,
        TransactionCategory.RESTAURANTS,
        TransactionCategory.TRANSPORTATION,
        TransactionCategory.SHOPPING,
        TransactionCategory.UTILITIES,
        TransactionCategory.SUBSCRIPTIONS,
        TransactionCategory.HEALTHCARE,
        TransactionCategory.ENTERTAINMENT,
    }
)


class ProductType(StrEnum):
    ACCOUNT = "account"
    CARD = "card"
    LOAN = "loan"
    MORTGAGE = "mortgage"
    DEPOSIT = "deposit"


class Channel(StrEnum):
    TEXT = "text"
    VOICE = "voice"


class SafetyStatus(StrEnum):
    ALLOWED = "allowed"
    REFUSED = "refused"
    CONFIRMATION_REQUIRED = "confirmation_required"


class EvaluationRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
