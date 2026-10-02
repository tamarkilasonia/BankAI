"""Evaluation endpoints (operator key, background runs) and the API-wide guarantees:
one error format, no internals, request IDs, health."""

import asyncio
import uuid

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from app.api.main import create_app
from tests.api.conftest import OPERATOR_KEY, Api, api_settings

OPERATOR = {"X-Operator-Key": OPERATOR_KEY}

# --- evaluations ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("headers", "status"), [({}, 403), ({"X-Operator-Key": "wrong" * 8}, 403)])
async def test_evaluations_need_the_operator_key(api: Api, headers: dict[str, str], status: int) -> None:
    for request in (
        api.client.get("/evaluations", headers=headers),
        api.client.post("/evaluations/run", json={}, headers=headers),
    ):
        response = await request
        assert response.status_code == status and response.json()["error"]["code"] == "forbidden"


async def test_evaluations_are_disabled_without_a_configured_key(api: Api) -> None:
    api.services.settings = api_settings(operator_api_key=None)
    response = await api.client.get("/evaluations", headers=OPERATOR)
    assert response.status_code == 503 and "OPERATOR_API_KEY" in response.json()["error"]["message"]


async def test_a_customer_session_is_not_an_operator(api: Api) -> None:
    assert (await api.client.get("/evaluations", headers=await api.login())).status_code == 403


async def test_run_in_the_background_then_read_the_results(api: Api) -> None:
    started = await api.client.post(
        "/evaluations/run", json={"categories": ["GENERAL"], "limit": 4}, headers=OPERATOR
    )

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert started.json() == {"run_id": run_id, "status": "running", "cases": 4}
    busy = await api.client.post("/evaluations/run", json={}, headers=OPERATOR)
    assert busy.status_code == 409  # one run at a time
    assert api.services.evaluation_task is not None
    await api.services.evaluation_task

    detail = (await api.client.get(f"/evaluations/{run_id}", headers=OPERATOR)).json()
    assert (detail["status"], detail["total_cases"], detail["llm_provider"]) == ("completed", 4, "none")
    assert len(detail["results"]) == 4 and detail["summary"]["metrics"]
    first = detail["results"][0]
    assert {
        "case_id",
        "input_text",
        "expected_behavior",
        "actual_behavior",
        "failure_reasons",
        "scores",
    } <= set(first)
    listed = (await api.client.get("/evaluations", headers=OPERATOR)).json()
    assert listed[0]["id"] == run_id and listed[0]["passed_cases"] == detail["passed_cases"]


@pytest.mark.parametrize(
    ("body", "message"),
    [({"categories": ["NOPE"]}, "Unknown category"), ({"tags": ["no-such-tag"]}, "No evaluation cases")],
)
async def test_bad_run_requests(api: Api, body: dict[str, object], message: str) -> None:
    response = await api.client.post("/evaluations/run", json=body, headers=OPERATOR)
    assert response.status_code == 422 and message in response.json()["error"]["message"]


async def test_unknown_run(api: Api) -> None:
    response = await api.client.get(f"/evaluations/{uuid.uuid4()}", headers=OPERATOR)
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


async def test_a_failing_background_run_is_marked_failed(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.evaluation import runner

    async def explode(*_: object, **__: object) -> None:
        raise RuntimeError("model server down")

    monkeypatch.setattr(runner, "run_cases", explode)
    run_id = (await api.client.post("/evaluations/run", json={"limit": 1}, headers=OPERATOR)).json()["run_id"]
    with pytest.raises(RuntimeError):
        await api.services.evaluation_task  # type: ignore[misc]
    await asyncio.sleep(0)  # let the done-callback log

    detail = (await api.client.get(f"/evaluations/{run_id}", headers=OPERATOR)).json()
    assert detail["status"] == "failed" and detail["summary"] == {"error": "RuntimeError: model server down"}


# --- errors, request IDs, health --------------------------------------------------------------------------


async def test_unexpected_errors_reveal_nothing(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(**_: object) -> None:
        raise ZeroDivisionError("secret internal detail at /app/agents/nodes.py:42")

    monkeypatch.setattr(api.services.assistant, "respond", explode)
    response = await api.client.post("/chat", json={"message": "hi"}, headers=await api.login())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert (
        "secret" not in response.text and "nodes.py" not in response.text and "Traceback" not in response.text
    )


async def test_database_outage_is_a_503(api: Api, nino_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def down() -> None:
        raise OperationalError("SELECT", {}, Exception("connection refused"))

    headers = await api.login()
    monkeypatch.setattr(api.services, "sessions", down)
    response = await api.client.get(f"/accounts/{nino_id}", headers=headers)

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "database_unavailable",
        "message": "Banking data is temporarily unavailable.",
        "request_id": response.headers["x-request-id"],
    }


@pytest.mark.parametrize(
    ("sent", "echoed"),
    [("gateway-trace-123", True), ("bad\nid", False), ("x" * 65, False)],
    ids=["valid", "newline", "too-long"],
)
async def test_request_ids(api: Api, sent: str, echoed: bool) -> None:
    response = await api.client.get("/health", headers={"X-Request-ID": sent})
    returned = response.headers["x-request-id"]
    assert (returned == sent) is echoed and len(returned) <= 64
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_unknown_routes_and_methods_use_the_error_format(api: Api) -> None:
    missing = await api.client.get("/nope")
    wrong_method = await api.client.get("/chat")
    assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")
    assert (wrong_method.status_code, wrong_method.json()["error"]["code"]) == (405, "method_not_allowed")


async def test_health_reports_components_without_secrets(api: Api) -> None:
    body = (await api.client.get("/health")).json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["components"]["tools"] == "mcp (in_process)" and body["components"]["llm"].startswith("none")
    assert "secret" not in str(body).lower()


async def test_openapi_documents_every_required_endpoint(api: Api) -> None:
    paths = (await api.client.get("/openapi.json")).json()["paths"]
    for path, method in [
        ("/chat", "post"),
        ("/voice/transcribe", "post"),
        ("/voice/synthesize", "post"),
        ("/accounts/{user_id}", "get"),
        ("/transactions/{user_id}", "get"),
        ("/evaluations", "get"),
        ("/evaluations/run", "post"),
        ("/health", "get"),
    ]:
        assert method in paths[path], path
    assert (await api.client.get("/docs")).status_code == 200


async def test_cors_is_off_unless_configured(monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:5173"]')
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        preflight = await client.options(
            "/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
        )
        other = await client.options(
            "/chat", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
        )
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers
