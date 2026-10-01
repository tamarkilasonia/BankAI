"""Account services and the authorization layer."""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.seed import generate_seed_data
from app.models.enums import AccountStatus, AccountType, Currency
from app.services.accounts import get_account_balance, get_account_details
from app.services.authorization import Principal, authorize_user_access, get_owned_account
from app.services.errors import AccessDeniedError
from tests.constants import (
    ANCHOR,
    EMMA,
    EMMA_FROZEN_EUR,
    GIORGI,
    GIORGI_CHECKING,
    NINO,
    NINO_CHECKING,
    NINO_SAVINGS_USD,
)

SEED = generate_seed_data(anchor=ANCHOR)


def expected_balances(user_id: uuid.UUID) -> dict[uuid.UUID, Decimal]:
    return {a.id: a.balance for a in SEED.accounts if a.user_id == user_id}


# --- authorize_user_access ----------------------------------------------------


def test_same_user_is_allowed() -> None:
    authorize_user_access(Principal(NINO), NINO)  # does not raise


def test_other_user_is_denied() -> None:
    with pytest.raises(AccessDeniedError, match="your own"):
        authorize_user_access(Principal(NINO), GIORGI)


async def test_get_owned_account_returns_own_account(seeded_session: AsyncSession) -> None:
    account = await get_owned_account(seeded_session, Principal(NINO), NINO_CHECKING)
    assert account.user_id == NINO


async def test_foreign_and_missing_accounts_are_indistinguishable(seeded_session: AsyncSession) -> None:
    with pytest.raises(AccessDeniedError) as foreign:
        await get_owned_account(seeded_session, Principal(NINO), GIORGI_CHECKING)
    with pytest.raises(AccessDeniedError) as missing:
        await get_owned_account(seeded_session, Principal(NINO), uuid.uuid4())

    assert foreign.value.message == missing.value.message  # no enumeration oracle


# --- balances -----------------------------------------------------------------


async def test_balance_lists_only_own_accounts(seeded_session: AsyncSession) -> None:
    result = await get_account_balance(seeded_session, Principal(NINO), now=ANCHOR)

    assert {a.account_id: a.balance for a in result.accounts} == expected_balances(NINO)
    assert result.as_of == ANCHOR
    assert "MOCK" in result.notice


async def test_totals_are_per_currency_never_mixed(seeded_session: AsyncSession) -> None:
    result = await get_account_balance(seeded_session, Principal(NINO), now=ANCHOR)

    gel = sum((a.balance for a in result.accounts if a.currency is Currency.GEL), Decimal("0"))
    assert result.totals_by_currency == {
        Currency.GEL: gel,
        Currency.USD: expected_balances(NINO)[NINO_SAVINGS_USD],
    }


@pytest.mark.parametrize(
    ("filters", "expected_count"),
    [
        ({"account_type": AccountType.SAVINGS}, 2),
        ({"currency": Currency.USD}, 1),
        ({"account_type": AccountType.CHECKING, "currency": Currency.GEL}, 1),
        ({"currency": Currency.EUR}, 0),
    ],
)
async def test_balance_filters(
    seeded_session: AsyncSession, filters: dict[str, object], expected_count: int
) -> None:
    result = await get_account_balance(seeded_session, Principal(NINO), now=ANCHOR, **filters)  # type: ignore[arg-type]

    assert len(result.accounts) == expected_count


async def test_balance_of_single_owned_account(seeded_session: AsyncSession) -> None:
    result = await get_account_balance(seeded_session, Principal(NINO), now=ANCHOR, account_id=NINO_CHECKING)

    assert [a.account_id for a in result.accounts] == [NINO_CHECKING]


async def test_balance_of_foreign_account_is_denied(seeded_session: AsyncSession) -> None:
    with pytest.raises(AccessDeniedError):
        await get_account_balance(seeded_session, Principal(NINO), now=ANCHOR, account_id=GIORGI_CHECKING)


async def test_frozen_account_reports_its_status(seeded_session: AsyncSession) -> None:
    result = await get_account_balance(
        seeded_session, Principal(EMMA), now=ANCHOR, account_id=EMMA_FROZEN_EUR
    )

    assert result.accounts[0].status is AccountStatus.FROZEN


# --- details ------------------------------------------------------------------


async def test_details_include_profile_and_accounts(seeded_session: AsyncSession) -> None:
    result = await get_account_details(seeded_session, Principal(NINO))

    assert (result.customer_number, result.full_name) == ("CUST-0001", "Nino Beridze")
    assert {a.account_id for a in result.accounts} == set(expected_balances(NINO))
    assert all(a.opened_at.tzinfo is not None for a in result.accounts)


async def test_details_for_unknown_customer_are_denied(seeded_session: AsyncSession) -> None:
    with pytest.raises(AccessDeniedError):
        await get_account_details(seeded_session, Principal(uuid.uuid4()))
