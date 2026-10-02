"""Account balances and details."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import as_utc
from app.models import Account, User
from app.models.enums import AccountType, Currency
from app.services.authorization import Principal, get_owned_account
from app.services.errors import AccessDeniedError
from app.services.schemas import AccountDetails, AccountDetailsResult, AccountSummary, BalanceResult


async def _owned_accounts(
    session: AsyncSession,
    principal: Principal,
    *,
    account_id: uuid.UUID | None = None,
    account_type: AccountType | None = None,
    currency: Currency | None = None,
) -> list[Account]:
    stmt = select(Account).where(Account.user_id == principal.user_id).order_by(Account.account_number)
    if account_id is not None:
        await get_owned_account(session, principal, account_id)  # explicit check, clear error
        stmt = stmt.where(Account.id == account_id)  # and the query stays scoped to the owner
    if account_type is not None:
        stmt = stmt.where(Account.account_type == account_type)
    if currency is not None:
        stmt = stmt.where(Account.currency == currency)
    return list(await session.scalars(stmt))


def _summary(account: Account) -> AccountSummary:
    return AccountSummary(
        account_id=account.id,
        account_number=account.account_number,
        name=account.name,
        account_type=account.account_type,
        currency=account.currency,
        balance=account.balance,
        status=account.status,
    )


async def get_account_balance(
    session: AsyncSession,
    principal: Principal,
    *,
    now: datetime,
    account_id: uuid.UUID | None = None,
    account_type: AccountType | None = None,
    currency: Currency | None = None,
) -> BalanceResult:
    """Balances of the customer's accounts, optionally filtered.

    ``account_id`` selects exactly one account (and must be owned); otherwise the
    type/currency filters apply to all of the customer's accounts.
    """
    accounts = await _owned_accounts(
        session, principal, account_id=account_id, account_type=account_type, currency=currency
    )
    totals: dict[Currency, Decimal] = {}
    for account in accounts:
        totals[account.currency] = totals.get(account.currency, Decimal("0.00")) + account.balance
    return BalanceResult(accounts=[_summary(a) for a in accounts], totals_by_currency=totals, as_of=now)


async def get_account_details(
    session: AsyncSession, principal: Principal, *, account_id: uuid.UUID | None = None
) -> AccountDetailsResult:
    user = await session.get(User, principal.user_id)
    if user is None:
        raise AccessDeniedError("Customer not found.")
    accounts = await _owned_accounts(session, principal, account_id=account_id)
    return AccountDetailsResult(
        customer_number=user.customer_number,
        full_name=user.full_name,
        accounts=[
            AccountDetails(**_summary(a).model_dump(), opened_at=as_utc(a.opened_at)) for a in accounts
        ],
    )
