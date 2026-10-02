"""GET /health - is the service up, can it reach its database, what is it configured with."""

from fastapi import APIRouter, Response, status

from app.api.dependencies import ServicesDep
from app.api.schemas import HealthResponse
from app.database.session import check_database, get_engine

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, services: ServicesDep) -> HealthResponse:
    """Liveness + database connectivity (503 when the database is unreachable). No secrets."""
    settings = services.settings
    engine = services.sessions.kw.get("bind") or get_engine()  # the database the services really use
    db_ok = await check_database(engine)
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        environment=settings.environment,
        database="ok" if db_ok else "unavailable",
        components={
            "llm": "none (rules + templates)" if settings.llm_provider == "none" else settings.llm_model,
            "embeddings": settings.embedding_model if settings.embedding_provider != "hashing" else "hashing",
            "tools": f"mcp ({settings.mcp_transport})",
            "stt": settings.stt_provider,
            "tts": settings.tts_provider,
        },
    )
