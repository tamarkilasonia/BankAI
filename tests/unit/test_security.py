"""Principal tokens: the agent -> MCP identity mechanism."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.security import (
    TOKEN_AUDIENCE,
    TOKEN_ISSUER,
    InvalidTokenError,
    create_principal_token,
    verify_principal_token,
)
from tests.constants import NINO, TEST_MCP_SECRET

OTHER_SECRET = "another-secret-that-is-long-enough-123456"


def claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "sub": str(NINO),
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(seconds=60),
    }
    return {k: v for k, v in (base | overrides).items() if v is not None}


def test_round_trip_returns_the_customer_id() -> None:
    token = create_principal_token(NINO, secret=TEST_MCP_SECRET)

    assert verify_principal_token(token, secret=TEST_MCP_SECRET) == NINO


def test_token_signed_with_another_secret_is_rejected() -> None:
    forged = create_principal_token(NINO, secret=OTHER_SECRET)

    with pytest.raises(InvalidTokenError):
        verify_principal_token(forged, secret=TEST_MCP_SECRET)


def test_expired_token_is_rejected() -> None:
    old = datetime.now(UTC) - timedelta(minutes=10)
    token = create_principal_token(NINO, secret=TEST_MCP_SECRET, ttl_seconds=60, now=old)

    with pytest.raises(InvalidTokenError, match="expired"):
        verify_principal_token(token, secret=TEST_MCP_SECRET)


def test_tampered_payload_breaks_the_signature() -> None:
    header, _, signature = create_principal_token(NINO, secret=TEST_MCP_SECRET).split(".")
    other_payload = jwt.encode(claims(sub=str(uuid.uuid4())), OTHER_SECRET, algorithm="HS256").split(".")[1]

    with pytest.raises(InvalidTokenError):
        verify_principal_token(f"{header}.{other_payload}.{signature}", secret=TEST_MCP_SECRET)


def test_unsigned_alg_none_token_is_rejected() -> None:
    unsigned = jwt.encode(claims(), key=None, algorithm="none")  # type: ignore[arg-type]

    with pytest.raises(InvalidTokenError):
        verify_principal_token(unsigned, secret=TEST_MCP_SECRET)


@pytest.mark.parametrize(
    "bad_claims",
    [
        {"aud": "some-other-service"},
        {"iss": "attacker"},
        {"sub": "not-a-uuid"},
        {"sub": None},  # missing subject
        {"exp": None},  # never expires
    ],
)
def test_wrong_or_missing_claims_are_rejected(bad_claims: dict[str, object]) -> None:
    token = jwt.encode(claims(**bad_claims), TEST_MCP_SECRET, algorithm="HS256")

    with pytest.raises(InvalidTokenError):
        verify_principal_token(token, secret=TEST_MCP_SECRET)


@pytest.mark.parametrize("token", [None, "", 123, "not.a.jwt"])
def test_missing_or_garbage_token_is_rejected(token: object) -> None:
    with pytest.raises(InvalidTokenError):
        verify_principal_token(token, secret=TEST_MCP_SECRET)
