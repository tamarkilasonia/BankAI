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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

Errors always look like `{"error": {"code", "message", "request_id"}}`; send the `request_id`
when reporting a problem. Evaluation endpoints need the `X-Operator-Key` header.
"""

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
    return app


app = create_app()
