"""FastAPI application: routers, middleware, error handling, OpenAPI docs.

    uvicorn app.api.main:app --reload       # then open http://localhost:8000/docs

Request flow::

    RequestContextMiddleware (request ID, access log, crash safety net)
      -> CORS (only if CORS_ORIGINS is set)
        -> route (pydantic validation, session token -> Principal)
          -> agent / services / voice pipeline
      <- errors become {"error": {"code", "message", "request_id"}}
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from app.api.errors import install_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routers import auth, banking, chat, evaluations, system, voice
from app.api.services import Services, build_services
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.session import get_engine

logger = get_logger(__name__)

DESCRIPTION = """
Agentic banking assistant prototype. **All banking data is MOCK data**; no money can ever move.

1. `POST /auth/login` with a mock customer number (`CUST-0001` ... `CUST-0010`) and use the
   `access_token` as a Bearer token (the **Authorize** button above).
2. `POST /chat` (text) or `POST /voice/chat` (audio) to talk to the agent.

The web UI is at [/ui/](/ui/).

Errors always look like `{"error": {"code", "message", "request_id"}}`; send the `request_id`
when reporting a problem. Evaluation endpoints need the `X-Operator-Key` header.
"""

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# The UI page: scripts, styles and requests only from this server; nothing inline, no frames.
# (Not applied to /docs, which loads Swagger UI from a CDN.)
FRONTEND_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-cache",
}


class FrontendFiles(StaticFiles):
    """The static UI, with the security headers above on every file."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(FRONTEND_HEADERS)
        return response


TAGS = [
    {"name": "auth", "description": "MOCK login: a customer number in, a signed session token out."},
    {"name": "chat", "description": "The LangGraph agent (tools via MCP, RAG, safety layer)."},
    {"name": "voice", "description": "Speech-to-text, text-to-speech and full spoken turns."},
    {"name": "banking data", "description": "The signed-in customer's own accounts and transactions."},
    {"name": "evaluations", "description": "Evaluation runs and results (operators only)."},
    {"name": "system", "description": "Health."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    injected = getattr(app.state, "services", None)  # tests inject their own
    services: Services = injected or await build_services(settings)
    app.state.services = services
    logger.info(
        "app.startup", extra={"environment": settings.environment, "database": settings.database_url_safe}
    )
    try:
        yield
    finally:
        if injected is None:
            await services.aclose()
            await get_engine().dispose()
        logger.info("app.shutdown")


def create_app(services: Services | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.8.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    if services is not None:
        app.state.services = services
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Operator-Key", "X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)  # added last = outermost: wraps everything
    install_error_handlers(app)
    for module in (auth, chat, voice, banking, evaluations, system):
        app.include_router(module.router)
    if FRONTEND_DIR.is_dir():
        app.mount("/ui", FrontendFiles(directory=FRONTEND_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def home() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app


app = create_app()
