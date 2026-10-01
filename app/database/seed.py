"""Deterministic MOCK data generator.

Everything produced here is fictional: names, e-mails (``@example.com`` is a
reserved domain), IBAN-shaped account numbers with an invalid check digit and
the bank code ``MOCK``, merchants and products.

Two properties matter for testing and evaluation:

1. **Determinism.** The same ``seed`` and ``anchor`` always produce identical
   data, and entity IDs are stable UUIDs (``uuid5``), so evaluation cases can
   refer to "customer CUST-0001" across re-seeds.
2. **Internal consistency.** Every account balance equals the sum of its
   ledger (credits - debits), and no account's running balance ever goes
   negative. The evaluation framework relies on this when it computes ground
   truth directly from SQL.

Generation (pure Python, easy to unit test) is separated from persistence
(``seed_database``), which needs a database session.
"""

import random
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import BANK_TZ
from app.models import Account, BankingProduct, Transaction, User
from app.models.enums import (
    AccountStatus,
    AccountType,
    Currency,
    Language,
    ProductType,
    TransactionCategory,
    TransactionDirection,
)

DEFAULT_SEED = 42
HISTORY_DAYS = 90

# Fixed namespace so uuid5 IDs are identical on every machine.
_SEED_NAMESPACE = uuid.UUID("6f1c2b0e-8a43-4d4b-9c51-5b2f0d3e7a10")
_CENT = Decimal("0.01")


def stable_uuid(*parts: object) -> uuid.UUID:
    """A UUID derived from its parts - same input, same UUID, every run."""
    return uuid.uuid5(_SEED_NAMESPACE, ":".join(str(part) for part in parts))


def mock_account_number(user_index: int, account_index: int) -> str:
    """IBAN-shaped (22 chars, like a Georgian IBAN) but deliberately invalid."""
    return f"GE00MOCK{user_index:06d}{account_index:08d}"


# ---------------------------------------------------------------------------
# Static mock catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserProfile:
    full_name: str
    language: Language
    employer: str
    monthly_salary: Decimal
    spending_level: float  # multiplies card spending amounts
    # Accounts besides the main GEL checking account: (type, currency, status)
    extra_accounts: tuple[tuple[AccountType, Currency, AccountStatus], ...] = ()


_ACTIVE = AccountStatus.ACTIVE

USER_PROFILES: tuple[UserProfile, ...] = (
    UserProfile(
        "Nino Beridze",
        Language.GEORGIAN,
        "Kartli Software LLC",
        Decimal("5200"),
        1.2,
        ((AccountType.SAVINGS, Currency.GEL, _ACTIVE), (AccountType.SAVINGS, Currency.USD, _ACTIVE)),
    ),
    UserProfile(
        "Giorgi Kapanadze",
        Language.GEORGIAN,
        "Mtkvari Logistics",
        Decimal("3400"),
        1.0,
        ((AccountType.SAVINGS, Currency.GEL, _ACTIVE),),
    ),
    UserProfile(
        "Ana Lomidze",
        Language.ENGLISH,
        "Caucasus Health Group",
        Decimal("4100"),
        0.9,
        ((AccountType.SAVINGS, Currency.EUR, _ACTIVE),),
    ),
    UserProfile(
        "Luka Maisuradze",
        Language.GEORGIAN,
        "Black Sea Shipping",
        Decimal("2600"),
        0.8,
        ((AccountType.CHECKING, Currency.USD, _ACTIVE),),
    ),
    UserProfile(
        "Mariam Tsiklauri",
        Language.ENGLISH,
        "Rioni Energy",
        Decimal("6300"),
        1.4,
        ((AccountType.SAVINGS, Currency.GEL, _ACTIVE), (AccountType.CHECKING, Currency.EUR, _ACTIVE)),
    ),
    UserProfile(
        "David Gelashvili",
        Language.ENGLISH,
        "Svaneti Tourism Co.",
        Decimal("2900"),
        0.9,
        ((AccountType.SAVINGS, Currency.USD, _ACTIVE),),
    ),
    UserProfile("Elene Chikovani", Language.GEORGIAN, "Tbilisi Public School No. 7", Decimal("1900"), 0.6),
    UserProfile(
        "Sandro Abashidze",
        Language.ENGLISH,
        "Kakheti Wine Exports",
        Decimal("3800"),
        1.1,
        ((AccountType.SAVINGS, Currency.GEL, _ACTIVE),),
    ),
    UserProfile("Irakli Natadze", Language.GEORGIAN, "Batumi Port Services", Decimal("3100"), 1.0),
    UserProfile(
        "Emma Novak",
        Language.ENGLISH,
        "Global Consulting Partners",
        Decimal("5800"),
        1.3,
        # One frozen account so status handling can be tested.
        ((AccountType.CHECKING, Currency.EUR, AccountStatus.FROZEN),),
    ),
)

MERCHANTS: dict[TransactionCategory, tuple[str, ...]] = {
    TransactionCategory.GROCERIES: ("FreshMart", "Daily Basket", "Green Valley Market", "Corner Grocer"),
    TransactionCategory.RESTAURANTS: (
        "Old Town Bistro",
        "Khinkali House",
        "Riverside Cafe",
        "Sushi Corner",
        "Pizza Lab",
    ),
    TransactionCategory.TRANSPORTATION: ("MetroCard Top-up", "City Taxi", "RideNow", "FuelPoint"),
    TransactionCategory.SHOPPING: ("TechZone", "StyleHub", "HomeGoods Depot", "BookNook"),
    TransactionCategory.HEALTHCARE: ("CarePlus Pharmacy", "MedCenter Clinic"),
    TransactionCategory.ENTERTAINMENT: ("StarCinema", "Concert Hall Tickets", "GameWorld"),
}


@dataclass(frozen=True)
class SpendingPattern:
    category: TransactionCategory
    events_per_month: float
    min_amount: Decimal
    max_amount: Decimal


# Card spending on the main GEL checking account.
GEL_SPENDING_PATTERNS: tuple[SpendingPattern, ...] = (
    SpendingPattern(TransactionCategory.GROCERIES, 9, Decimal("15"), Decimal("140")),
    SpendingPattern(TransactionCategory.RESTAURANTS, 5, Decimal("25"), Decimal("160")),
    SpendingPattern(TransactionCategory.TRANSPORTATION, 8, Decimal("2"), Decimal("45")),
    SpendingPattern(TransactionCategory.SHOPPING, 2, Decimal("40"), Decimal("420")),
    SpendingPattern(TransactionCategory.HEALTHCARE, 1, Decimal("15"), Decimal("180")),
    SpendingPattern(TransactionCategory.ENTERTAINMENT, 1.5, Decimal("20"), Decimal("120")),
)

# Occasional card spending abroad on USD/EUR checking accounts.
FX_SPENDING_PATTERNS: tuple[SpendingPattern, ...] = (
    SpendingPattern(TransactionCategory.RESTAURANTS, 1.5, Decimal("12"), Decimal("90")),
    SpendingPattern(TransactionCategory.SHOPPING, 1, Decimal("20"), Decimal("250")),
    SpendingPattern(TransactionCategory.TRANSPORTATION, 1, Decimal("5"), Decimal("60")),
)

# (merchant, day of month, min, max) - billed once a month.
UTILITY_BILLS: tuple[tuple[str, int, Decimal, Decimal], ...] = (
    ("City Power Co.", 5, Decimal("45"), Decimal("120")),
    ("AquaCity Water", 8, Decimal("12"), Decimal("30")),
    ("GasNet", 10, Decimal("20"), Decimal("140")),
    ("FiberLink Internet", 15, Decimal("45"), Decimal("45")),
)

# (merchant, day of month, fixed monthly price)
SUBSCRIPTIONS: tuple[tuple[str, int, Decimal], ...] = (
    ("StreamFlix", 3, Decimal("24.99")),
    ("MusicBox Premium", 7, Decimal("12.99")),
    ("CloudDrive Plus", 12, Decimal("8.50")),
    ("FitLife Gym", 1, Decimal("95.00")),
    ("NewsDaily Digital", 20, Decimal("15.00")),
)

SALARY_DAY = 25
SAVINGS_TRANSFER_DAY = 26

# MOCK product catalogue. These numbers are also what the knowledge base
# documents (phase 4) must state, so RAG answers and calculator results agree.
PRODUCTS: tuple[dict[str, object], ...] = (
    {
        "code": "ACC_CURRENT",
        "name": "Current Account",
        "product_type": ProductType.ACCOUNT,
        "description": "Everyday account for payments and salary. MOCK product.",
        "monthly_fee": Decimal("0.00"),
    },
    {
        "code": "ACC_SAVINGS",
        "name": "Flexible Savings Account",
        "product_type": ProductType.DEPOSIT,
        "description": "Savings account with withdrawals at any time. MOCK product.",
        "interest_rate_pct": Decimal("7.50"),
        "monthly_fee": Decimal("0.00"),
    },
    {
        "code": "DEP_TERM_12M",
        "name": "12-Month Term Deposit",
        "product_type": ProductType.DEPOSIT,
        "description": "Fixed-term deposit locked for 12 months. MOCK product.",
        "interest_rate_pct": Decimal("10.00"),
        "min_amount": Decimal("1000.00"),
        "min_term_months": 12,
        "max_term_months": 12,
    },
    {
        "code": "CARD_DEBIT_CLASSIC",
        "name": "Classic Debit Card",
        "product_type": ProductType.CARD,
        "description": "Debit card linked to a current account. MOCK product.",
        "monthly_fee": Decimal("0.00"),
    },
    {
        "code": "CARD_DEBIT_GOLD",
        "name": "Gold Debit Card",
        "product_type": ProductType.CARD,
        "description": "Debit card with travel insurance and higher limits. MOCK product.",
        "monthly_fee": Decimal("5.00"),
    },
    {
        "code": "CARD_CREDIT_STANDARD",
        "name": "Standard Credit Card",
        "product_type": ProductType.CARD,
        "description": "Revolving credit card with up to 55 interest-free days. MOCK product.",
        "interest_rate_pct": Decimal("24.00"),
        "monthly_fee": Decimal("2.00"),
        "max_amount": Decimal("10000.00"),
    },
    {
        "code": "LOAN_CONSUMER",
        "name": "Consumer Loan",
        "product_type": ProductType.LOAN,
        "description": "Unsecured personal loan. MOCK product.",
        "interest_rate_pct": Decimal("14.50"),
        "min_amount": Decimal("500.00"),
        "max_amount": Decimal("50000.00"),
        "min_term_months": 6,
        "max_term_months": 60,
    },
    {
        "code": "LOAN_AUTO",
        "name": "Auto Loan",
        "product_type": ProductType.LOAN,
        "description": "Loan for buying a new or used car. MOCK product.",
        "interest_rate_pct": Decimal("11.90"),
        "min_amount": Decimal("5000.00"),
        "max_amount": Decimal("150000.00"),
        "min_term_months": 12,
        "max_term_months": 84,
    },
    {
        "code": "MORTGAGE_STANDARD",
        "name": "Standard Mortgage",
        "product_type": ProductType.MORTGAGE,
        "description": "Home loan secured by the purchased property. MOCK product.",
        "interest_rate_pct": Decimal("9.50"),
        "min_amount": Decimal("20000.00"),
        "max_amount": Decimal("1000000.00"),
        "min_term_months": 60,
        "max_term_months": 360,
    },
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class SeedData:
    users: list[User] = field(default_factory=list)
    accounts: list[Account] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    products: list[BankingProduct] = field(default_factory=list)


@dataclass(frozen=True)
class SeedSummary:
    users: int
    accounts: int
    transactions: int
    products: int
    anchor: str


@dataclass
class _LedgerEntry:
    """A transaction before it becomes an ORM object."""

    booked_at: datetime
    direction: TransactionDirection
    amount: Decimal
    category: TransactionCategory
    merchant: str
    description: str

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction is TransactionDirection.CREDIT else -self.amount


class _Window:
    """The mock history period: (anchor - HISTORY_DAYS, anchor]."""

    def __init__(self, anchor: datetime) -> None:
        self.end = anchor
        self.start = anchor - timedelta(days=HISTORY_DAYS)

    def contains(self, moment: datetime) -> bool:
        return self.start < moment <= self.end

    def months(self) -> Iterator[tuple[int, int]]:
        """(year, month) for every calendar month overlapping the window."""
        first = self.start.astimezone(BANK_TZ).date().replace(day=1)
        last = self.end.astimezone(BANK_TZ).date()
        year, month = first.year, first.month
        while (year, month) <= (last.year, last.month):
            yield year, month
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    def monthly(self, day: int, at: time) -> Iterator[datetime]:
        """The given day/time in each month, if it falls inside the window."""
        for year, month in self.months():
            moment = datetime.combine(date(year, month, day), at, tzinfo=BANK_TZ).astimezone(UTC)
            if self.contains(moment):
                yield moment


def _money(rng: random.Random, low: Decimal, high: Decimal, scale: float = 1.0) -> Decimal:
    cents = rng.randint(int(low * 100), int(high * 100))
    return (Decimal(cents) / 100 * Decimal(str(scale))).quantize(_CENT)


def _random_moment(rng: random.Random, window: _Window) -> datetime:
    """A random daytime moment (08:00-21:59 bank time) inside the window."""
    local_start = window.start.astimezone(BANK_TZ).date()
    day = local_start + timedelta(days=rng.randint(1, HISTORY_DAYS))
    at = time(rng.randint(8, 21), rng.randint(0, 59), rng.randint(0, 59))
    return datetime.combine(day, at, tzinfo=BANK_TZ).astimezone(UTC)


def _card_spending(
    rng: random.Random, window: _Window, patterns: tuple[SpendingPattern, ...], scale: float
) -> list[_LedgerEntry]:
    entries = []
    months = HISTORY_DAYS / 30
    for pattern in patterns:
        expected = pattern.events_per_month * months
        count = max(0, round(rng.gauss(expected, expected * 0.15)))
        for _ in range(count):
            moment = _random_moment(rng, window)
            if not window.contains(moment):
                continue
            merchant = rng.choice(MERCHANTS[pattern.category])
            entries.append(
                _LedgerEntry(
                    booked_at=moment,
                    direction=TransactionDirection.DEBIT,
                    amount=_money(rng, pattern.min_amount, pattern.max_amount, scale),
                    category=pattern.category,
                    merchant=merchant,
                    description=f"Card payment - {merchant}",
                )
            )
    return entries


def _opening_balance(rng: random.Random, entries: list[_LedgerEntry]) -> Decimal:
    """Smallest comfortable opening balance that keeps the account >= 0 at all times."""
    running = lowest = Decimal("0")
    for entry in sorted(entries, key=lambda e: e.booked_at):
        running += entry.signed_amount
        lowest = min(lowest, running)
    buffer = _money(rng, Decimal("150"), Decimal("1500"))
    return max(_money(rng, Decimal("300"), Decimal("2500")), -lowest + buffer)


def generate_seed_data(anchor: datetime | None = None, seed: int = DEFAULT_SEED) -> SeedData:
    """Build the complete mock dataset in memory (no database access).

    Args:
        anchor: The "now" of the mock history; transactions cover the
            ``HISTORY_DAYS`` before it. Defaults to the current time.
        seed: Random seed. Same seed + same anchor -> identical data.
    """
    if anchor is not None and anchor.tzinfo is None:
        raise ValueError("anchor must be timezone-aware (e.g. datetime.now(UTC))")
    anchor = (anchor or datetime.now(UTC)).astimezone(UTC)
    rng = random.Random(seed)
    window = _Window(anchor)
    data = SeedData()

    for row in PRODUCTS:
        data.products.append(BankingProduct(id=stable_uuid("product", row["code"]), **row))

    for user_index, profile in enumerate(USER_PROFILES, start=1):
        first, last = profile.full_name.lower().split()
        user = User(
            id=stable_uuid("user", user_index),
            customer_number=f"CUST-{user_index:04d}",
            full_name=profile.full_name,
            email=f"{first}.{last}@example.com",
            preferred_language=profile.language,
        )
        data.users.append(user)

        specs = ((AccountType.CHECKING, Currency.GEL, AccountStatus.ACTIVE), *profile.extra_accounts)
        accounts: list[Account] = []
        ledgers: list[list[_LedgerEntry]] = []
        for account_index, (account_type, currency, status) in enumerate(specs, start=1):
            accounts.append(
                Account(
                    id=stable_uuid("account", user_index, account_index),
                    user_id=user.id,
                    account_number=mock_account_number(user_index, account_index),
                    name=f"{account_type.value.title()} {currency.value}",
                    account_type=account_type,
                    currency=currency,
                    status=status,
                    opened_at=window.start - timedelta(days=rng.randint(200, 2500)),
                )
            )
            ledgers.append([])

        main_ledger = ledgers[0]

        # Salary, monthly bills and subscriptions on the main GEL account.
        for moment in window.monthly(SALARY_DAY, time(10, 0)):
            main_ledger.append(
                _LedgerEntry(
                    moment,
                    TransactionDirection.CREDIT,
                    profile.monthly_salary,
                    TransactionCategory.SALARY,
                    profile.employer,
                    f"Salary payment - {profile.employer}",
                )
            )
        for merchant, day, low, high in UTILITY_BILLS:
            for moment in window.monthly(day, time(9, 30)):
                main_ledger.append(
                    _LedgerEntry(
                        moment,
                        TransactionDirection.DEBIT,
                        _money(rng, low, high),
                        TransactionCategory.UTILITIES,
                        merchant,
                        f"Utility bill - {merchant}",
                    )
                )
        for merchant, day, price in rng.sample(SUBSCRIPTIONS, k=rng.randint(1, 3)):
            for moment in window.monthly(day, time(6, 0)):
                main_ledger.append(
                    _LedgerEntry(
                        moment,
                        TransactionDirection.DEBIT,
                        price,
                        TransactionCategory.SUBSCRIPTIONS,
                        merchant,
                        f"Subscription - {merchant}",
                    )
                )
        main_ledger.extend(_card_spending(rng, window, GEL_SPENDING_PATTERNS, profile.spending_level))

        # Secondary accounts.
        for account, ledger in zip(accounts[1:], ledgers[1:], strict=True):
            if account.account_type is AccountType.SAVINGS and account.currency is Currency.GEL:
                # Monthly transfer from checking -> savings, recorded on both sides.
                for moment in window.monthly(SAVINGS_TRANSFER_DAY, time(11, 0)):
                    amount = (profile.monthly_salary * Decimal(str(rng.uniform(0.08, 0.15)))).quantize(_CENT)
                    suffix = account.account_number[-4:]
                    main_ledger.append(
                        _LedgerEntry(
                            moment,
                            TransactionDirection.DEBIT,
                            amount,
                            TransactionCategory.TRANSFER,
                            "Own account transfer",
                            f"Transfer to savings ****{suffix}",
                        )
                    )
                    ledger.append(
                        _LedgerEntry(
                            moment,
                            TransactionDirection.CREDIT,
                            amount,
                            TransactionCategory.TRANSFER,
                            "Own account transfer",
                            "Transfer from checking",
                        )
                    )
            elif account.account_type is AccountType.SAVINGS:
                for moment in window.monthly(3, time(12, 0)):
                    ledger.append(
                        _LedgerEntry(
                            moment,
                            TransactionDirection.CREDIT,
                            _money(rng, Decimal("100"), Decimal("500")),
                            TransactionCategory.DEPOSIT,
                            "Branch cash desk",
                            f"Cash deposit ({account.currency.value})",
                        )
                    )
            else:
                ledger.extend(_card_spending(rng, window, FX_SPENDING_PATTERNS, 1.0))

        # Opening balance at the start of the window, then persist the ledger.
        for account, ledger in zip(accounts, ledgers, strict=True):
            opening = _opening_balance(rng, ledger)
            ledger.append(
                _LedgerEntry(
                    window.start,
                    TransactionDirection.CREDIT,
                    opening,
                    TransactionCategory.DEPOSIT,
                    "Opening balance",
                    "Balance at the start of the mock history",
                )
            )
            ledger.sort(key=lambda e: e.booked_at)
            account.balance = sum((e.signed_amount for e in ledger), Decimal("0")).quantize(_CENT)
            for tx_index, entry in enumerate(ledger, start=1):
                data.transactions.append(
                    Transaction(
                        id=stable_uuid("transaction", account.id, tx_index),
                        account_id=account.id,
                        direction=entry.direction,
                        amount=entry.amount,
                        currency=account.currency,
                        category=entry.category,
                        merchant=entry.merchant,
                        description=entry.description,
                        booked_at=entry.booked_at,
                    )
                )
        data.accounts.extend(accounts)

    return data


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class SeedError(RuntimeError):
    """Raised when the database is not in a state that can be seeded."""


async def clear_mock_data(session: AsyncSession) -> None:
    """Delete all users, accounts, transactions and products.

    Conversation logs keep their rows (their ``user_id`` becomes NULL through
    ``ON DELETE SET NULL``); evaluation results are untouched.
    """
    for model in (Transaction, Account, BankingProduct, User):
        await session.execute(delete(model))


async def seed_database(
    session: AsyncSession,
    *,
    anchor: datetime | None = None,
    seed: int = DEFAULT_SEED,
    reset: bool = False,
) -> SeedSummary:
    """Insert the mock dataset in a single transaction.

    Refuses to run on a database that already has users unless ``reset`` is
    True, so it can never silently duplicate or mix data.
    """
    if reset:
        await clear_mock_data(session)
    elif await session.scalar(select(func.count()).select_from(User)):
        raise SeedError("Database already contains users. Re-run with --reset to replace the mock data.")

    anchor = anchor or datetime.now(UTC)
    data = generate_seed_data(anchor=anchor, seed=seed)
    session.add_all([*data.users, *data.products, *data.accounts, *data.transactions])
    await session.commit()

    return SeedSummary(
        users=len(data.users),
        accounts=len(data.accounts),
        transactions=len(data.transactions),
        products=len(data.products),
        anchor=anchor.isoformat(),
    )
