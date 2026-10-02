"""MOCK login: a customer number in, a signed session token out.

There are no passwords - this is a prototype with MOCK customers. What *is* real is
everything after login: the token is signed (HS256), expires, and every request's
identity comes from it alone.
"""

from fastapi import APIRouter
from sqlalchemy import select

from app.api.dependencies import PrincipalDep, ServicesDep
from app.api.errors import ApiError
from app.api.schemas import CustomerProfile, ErrorResponse, LoginRequest, LoginResponse
from app.core.security import create_session_token
from app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


def _profile(user: User) -> CustomerProfile:
    return CustomerProfile(
        user_id=user.id,
        customer_number=user.customer_number,
        full_name=user.full_name,
        preferred_language=user.preferred_language.value,
    )


@router.post("/login", responses={401: {"model": ErrorResponse}})
async def login(body: LoginRequest, services: ServicesDep) -> LoginResponse:
    """Sign in as a MOCK customer (CUST-0001 ... CUST-0010)."""
    async with services.sessions() as session:
        user = await session.scalar(select(User).where(User.customer_number == body.customer_number.upper()))
    if user is None:
        raise ApiError(401, "unauthenticated", "Sign-in failed.")
    ttl = services.settings.api_session_ttl_minutes * 60
    token = create_session_token(user.id, secret=services.session_secret, ttl_seconds=ttl)
    return LoginResponse(access_token=token, expires_in=ttl, customer=_profile(user))


@router.get("/me", responses={401: {"model": ErrorResponse}})
async def me(principal: PrincipalDep, services: ServicesDep) -> CustomerProfile:
    """The signed-in customer."""
    async with services.sessions() as session:
        user = await session.get(User, principal.user_id)
    if user is None:  # the customer was deleted while the session was alive
        raise ApiError(401, "unauthenticated", "Please sign in.")
    return _profile(user)
