"""FastAPI application factory.

Phase 1 only exposes ``/health``. Routers for chat, voice, accounts and
evaluations are added in phase 8.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.session import check_database, get_engine

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    environment: str
    database: Literal["ok", "unavailable"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "app.startup",
        extra={"environment": settings.environment, "database": settings.database_url_safe},
    )
    yield
    await get_engine().dispose()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Agentic banking assistant prototype. All banking data is MOCK data.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(response: Response) -> HealthResponse:
        """Liveness + database connectivity. Returns 503 when the database is unreachable."""
        db_ok = await check_database(get_engine())
        if not db_ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            environment=settings.environment,
            database="ok" if db_ok else "unavailable",
        )

    return app


app = create_app()
