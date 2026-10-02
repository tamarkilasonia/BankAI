import httpx
import pytest

from app.api.main import create_app


async def get_health() -> httpx.Response:
    """The real startup path: settings from the environment, services built in the lifespan."""
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")


@pytest.mark.usefixtures("reset_app_state")
async def test_health_ok_when_database_reachable() -> None:
    response = await get_health()

    assert response.status_code == 200
    assert response.json() | {"components": {}} == {
        "status": "ok",
        "environment": "test",
        "database": "ok",
        "components": {},
    }
    assert response.json()["components"]["llm"] == "none (rules + templates)"


@pytest.mark.usefixtures("reset_app_state")
async def test_health_degraded_when_database_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:////nonexistent-dir/bankai.db")

    response = await get_health()

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
    assert response.json()["status"] == "degraded"


@pytest.mark.usefixtures("reset_app_state")
async def test_lifespan_starts_and_stops_cleanly() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        pass


def test_openapi_schema_is_generated() -> None:
    schema = create_app().openapi()
    assert "/health" in schema["paths"]
    assert "MOCK" in schema["info"]["description"]
