"""Banking data for the UI: GET /accounts/{user_id}, GET /transactions/{user_id}.

These endpoints call the domain services directly: the web UI is a first-party client
of the backend, while MCP is the boundary for the *agent's* tools. Authorization is the
same either way - it lives in the service layer:

1. the path's ``user_id`` must be the signed-in customer (``authorize_user_access``) -
   asking for another customer is a 403, identical whether that customer exists or not;
2. every service query is scoped to the principal anyway (defence in depth).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import PrincipalDep, ServicesDep
from app.api.schemas import AccountsResponse, ErrorResponse, TransactionsQuery
from app.services.accounts import get_account_balance, get_account_details
from app.services.authorization import authorize_user_access
from app.services.periods import resolve_period
from app.services.schemas import TransactionList
from app.services.transactions import get_transactions

router = APIRouter(tags=["banking data"])
ERRORS = {code: {"model": ErrorResponse} for code in (401, 403, 422, 503)}


@router.get("/accounts/{user_id}", responses=ERRORS)  # type: ignore[arg-type]
async def accounts(user_id: uuid.UUID, principal: PrincipalDep, services: ServicesDep) -> AccountsResponse:
    """The customer's accounts with balances (MOCK data). Only your own ``user_id`` is allowed."""
    authorize_user_access(principal, user_id)
    async with services.sessions() as session:
        details = await get_account_details(session, principal)
        balances = await get_account_balance(session, principal, now=services.assistant.deps.clock())
    return AccountsResponse(
        customer_number=details.customer_number,
        full_name=details.full_name,
        accounts=details.accounts,
        totals_by_currency={currency.value: total for currency, total in balances.totals_by_currency.items()},
    )


@router.get("/transactions/{user_id}", responses=ERRORS)  # type: ignore[arg-type]
async def transactions(
    user_id: uuid.UUID,
    query: Annotated[TransactionsQuery, Query()],
    principal: PrincipalDep,
    services: ServicesDep,
) -> TransactionList:
    """The customer's transactions, newest first (MOCK data). Only your own ``user_id`` is allowed."""
    authorize_user_access(principal, user_id)
    # The agent's clock, so the data endpoints and the chat agree on what "this month" is.
    now = services.assistant.deps.clock()
    window = resolve_period(query.period, now=now, start_date=query.start_date, end_date=query.end_date)
    async with services.sessions() as session:
        return await get_transactions(
            session,
            principal,
            date_range=window,
            category=query.category,
            direction=query.direction,
            limit=query.limit,
        )
