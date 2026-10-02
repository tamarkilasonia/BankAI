"""One error format for the whole API, and no internals in it.

Every error response looks like::

    {"error": {"code": "access_denied", "message": "...", "request_id": "3f2a..."}}

* ``code`` is stable and machine-readable (clients branch on it);
* ``message`` is written for people and never contains stack traces, SQL, file paths
  or the rejected input values;
* ``request_id`` lets support find the matching log lines.

Expected problems (bad input, wrong customer, bad audio, unavailable services) map to
4xx/503. Anything unexpected becomes a generic 500 - the details go to the log only.
"""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var
from app.services.errors import AccessDeniedError, BusinessRuleError, DomainError, NotFoundError
from app.voice.errors import VoiceError

logger = get_logger(__name__)

INTERNAL_ERROR_MESSAGE = "Something went wrong on our side. Please try again."

VOICE_STATUS = {
    "invalid_audio": status.HTTP_400_BAD_REQUEST,
    "unsupported_audio": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "audio_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "audio_too_long": status.HTTP_413_CONTENT_TOO_LARGE,
    "unsupported_language": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "text_too_long": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "voice_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "stt_failed": status.HTTP_502_BAD_GATEWAY,
    "tts_failed": status.HTTP_502_BAD_GATEWAY,
}
HTTP_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    503: "unavailable",
}


class ApiError(Exception):
    """An expected error with a status, a stable code and a customer-facing message."""

    def __init__(
        self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        error["details"] = details
    return {"error": error}


def error_response(
    status_code: int, code: str, message: str, *, details: Any = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(error_body(code, message, details), status_code=status_code, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, headers=exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTP_CODES.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else code.replace("_", " ").capitalize()
        return error_response(exc.status_code, code, message, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field and problem only - never the submitted value (it may be personal data).
        details = [
            {"field": ".".join(str(part) for part in error["loc"]), "problem": error["msg"]}
            for error in exc.errors()
        ]
        return error_response(422, "invalid_request", "The request is not valid.", details=details)

    @app.exception_handler(VoiceError)
    async def voice_error(_: Request, exc: VoiceError) -> JSONResponse:
        return error_response(VOICE_STATUS.get(exc.code, 500), exc.code, exc.message)

    @app.exception_handler(DomainError)
    async def domain_error(_: Request, exc: DomainError) -> JSONResponse:
        if isinstance(exc, AccessDeniedError):
            return error_response(403, exc.code, "You can only access your own banking information.")
        if isinstance(exc, NotFoundError):
            return error_response(404, exc.code, exc.message)
        if isinstance(exc, BusinessRuleError):
            return error_response(422, exc.code, exc.message)
        return error_response(503, exc.code, "This service is temporarily unavailable.")

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("http.database_error", extra={"error": type(exc).__name__})
        return error_response(503, "database_unavailable", "Banking data is temporarily unavailable.")
