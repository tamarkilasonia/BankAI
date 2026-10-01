"""The authorization layer: customers may only access their own data.

Every service that reads personal data takes a ``Principal`` (the authenticated
customer) and enforces ownership here - not in the agent, not in the prompt.
This is the security boundary of the whole system: even if a prompt injection
fools the LLM, the tool layer still refuses to return another customer's data.

Two mechanisms work together:
1. **Explicit checks** (this module) when a specific resource is requested,
   e.g. ``account_id=...``: raise ``AccessDeniedError`` if it is not owned.
2. **Query scoping** in the services: every query joins on
   ``accounts.user_id = principal.user_id``, so a forgotten check still cannot
   leak rows.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import Account
from app.services.errors import AccessDeniedError

logger = get_logger(__name__)


@dataclass(frozen=True)
class Principal:
    """The authenticated customer on whose behalf a request runs."""

    user_id: uuid.UUID


def authorize_user_access(principal: Principal, requested_user_id: uuid.UUID) -> None:
    """Allow access only when the requested customer is the authenticated one.

    There is currently no operation that legitimately crosses customers (no
    joint accounts, no bank-employee role); if one is added it belongs here as
    an explicit, tested rule.
    """
    if principal.user_id != requested_user_id:
        logger.warning(
            "authz.denied",
            extra={"principal": str(principal.user_id), "requested_user": str(requested_user_id)},
        )
        raise AccessDeniedError("You can only access your own banking information.")


async def get_owned_account(session: AsyncSession, principal: Principal, account_id: uuid.UUID) -> Account:
    """Load an account if - and only if - the principal owns it."""
    account = await session.get(Account, account_id)
    if account is None or account.user_id != principal.user_id:
        # Same error for "missing" and "someone else's", but the log knows which.
        logger.warning(
            "authz.denied",
            extra={
                "principal": str(principal.user_id),
                "account_id": str(account_id),
                "reason": "not_found" if account is None else "not_owner",
            },
        )
        raise AccessDeniedError("Account not found or not accessible.")
    return account
