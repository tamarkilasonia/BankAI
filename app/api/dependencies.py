"""FastAPI dependencies: the services, the signed-in customer, the operator key.

Identity comes from the signed session token only - never from the URL, the body or
the LLM. Endpoints that name a customer in the path (``/accounts/{user_id}``) still
check it against this identity (``authorize_user_access``).
"""

import secrets
from typing import Annotated

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.errors import ApiError
from app.api.services import Services
from app.core.security import InvalidTokenError, verify_session_token
from app.services.authorization import Principal
from app.voice.errors import SpeechUnavailableError
from app.voice.pipeline import VoicePipeline

_bearer = HTTPBearer(auto_error=False, description="The access_token from POST /auth/login (MOCK login).")


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_principal(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    unauthenticated = ApiError(
        401, "unauthenticated", "Please sign in.", headers={"WWW-Authenticate": "Bearer"}
    )
    if credentials is None:
        raise unauthenticated
    try:
        return Principal(verify_session_token(credentials.credentials, secret=services.session_secret))
    except InvalidTokenError:
        raise unauthenticated from None


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def require_operator(
    services: ServicesDep, key: Annotated[str | None, Header(alias="X-Operator-Key")] = None
) -> None:
    """Evaluation endpoints: operators only (they cost compute and show every mock customer's answers)."""
    configured = services.settings.operator_api_key
    if configured is None:
        raise ApiError(503, "unavailable", "Evaluation endpoints are disabled (OPERATOR_API_KEY is not set).")
    if key is None or not secrets.compare_digest(key.encode(), configured.get_secret_value().encode()):
        raise ApiError(403, "forbidden", "A valid X-Operator-Key header is required.")


OperatorDep = Depends(require_operator)


def get_voice(services: ServicesDep) -> VoicePipeline:
    voice = services.voice
    if voice is None:
        raise SpeechUnavailableError("Voice input is not enabled on this server (STT_PROVIDER).")
    return voice


VoiceDep = Annotated[VoicePipeline, Depends(get_voice)]
