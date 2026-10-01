"""Principal tokens: how the agent proves to the MCP server *which customer* it acts for.

Problem: the agent calls MCP tools on behalf of a logged-in customer. If the
customer's ID were an ordinary tool argument, the LLM could fill it in - and a
prompt injection ("use user_id=...") could change it.

Solution: the backend (which authenticated the customer) signs a short-lived
JWT whose subject is the customer ID and attaches it to the MCP request's
``_meta`` field - outside the tool arguments the LLM can see or produce. The
MCP server verifies the signature and takes the customer ID from the token only.

HS256 with a shared secret is enough for two services we operate ourselves.
With third-party clients you would use asymmetric keys (RS256/ES256) or the
OAuth 2.1 flow from the MCP authorization spec.
"""

import uuid
from datetime import datetime, timedelta

import jwt

from app.core.time import utcnow

# Public identifiers, not secrets (the secret comes from MCP_AUTH_SECRET).
PRINCIPAL_META_KEY = "bankai/principal_token"
TOKEN_ISSUER = "bankai-agent"  # noqa: S105
TOKEN_AUDIENCE = "bankai-mcp"  # noqa: S105
_ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired, or not signed by us."""


def create_principal_token(
    user_id: uuid.UUID, *, secret: str, ttl_seconds: int = 60, now: datetime | None = None
) -> str:
    issued_at = now or utcnow()
    claims = {
        "sub": str(user_id),
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "iat": issued_at,
        "exp": issued_at + timedelta(seconds=ttl_seconds),
    }
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


def verify_principal_token(token: object, *, secret: str) -> uuid.UUID:
    """Return the customer ID the token was issued for, or raise InvalidTokenError."""
    if not isinstance(token, str) or not token:
        raise InvalidTokenError("missing principal token")
    try:
        claims = jwt.decode(
            token,
            secret,
            # Pinning the algorithm blocks "alg: none" and algorithm-confusion attacks.
            algorithms=[_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
            options={"require": ["sub", "iss", "aud", "iat", "exp"]},
        )
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidTokenError(str(exc)) from exc
