"""Request context: a request ID on every request, an access log line, a safety net for crashes.

Written as plain ASGI middleware (not ``BaseHTTPMiddleware``) so the request ID,
stored in a ``ContextVar``, is visible to everything that runs for the request -
route handlers, the agent, the MCP client - and travels on to the MCP server.
"""

import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import INTERNAL_ERROR_MESSAGE, error_response
from app.core.logging import get_logger, request_id_var, safe_request_id

logger = get_logger("app.api")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # A caller may pass its own ID (e.g. a gateway); it is accepted only if it looks like one.
        request_id = safe_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER)) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        response_status: list[int] = []

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_status.append(message["status"])
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            # Last line of defence: log everything, tell the client nothing internal.
            logger.exception("http.unhandled_error")
            if not response_status:
                response = error_response(500, "internal_error", INTERNAL_ERROR_MESSAGE)
                await response(scope, receive, send_with_headers)
        finally:
            logger.info(
                "http.request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],  # no query string: it can carry personal filters
                    "status": response_status[0] if response_status else None,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            request_id_var.reset(token)
