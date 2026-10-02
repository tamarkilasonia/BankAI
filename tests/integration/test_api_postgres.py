"""FastAPI + PostgreSQL: the HTTP API on the real database engine.

``tests/api`` covers every endpoint on fast in-memory SQLite. This file repeats the
paths where the engine matters on PostgreSQL itself: NUMERIC money over JSON, timestamptz
period filters, the audit trail, and evaluation results stored as JSONB (which reorders
object keys - the dashboard's category order once depended on that).
"""

import asyncio
import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.agents.assistant import BankingAssistant
from app.agents.nodes import AgentDeps
from app.api.conversations import ConversationStore
from app.api.main import create_app
from app.api.services import Services
from app.database.seed import seed_database
from app.mcp.client import MCPToolGateway
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.models import ConversationLog, EvaluationResult
from app.rag.retriever import Retriever
from app.services.periods import Period, resolve_period
from tests.api.conftest import OPERATOR_KEY, SESSION_SECRET, Api, api_settings
from tests.api.test_concurrency import CUSTOMERS, QUESTIONS, markers
from tests.constants import ANCHOR, GIORGI, NINO, TEST_MCP_SECRET
from tests.fakes import FakeSTT, FakeTTS
from tests.oracle import SEED, user_transactions

pytestmark = pytest.mark.postgres
OPERATOR = {"X-Operator-Key": OPERATOR_KEY}


@pytest.fixture
async def pg_api(pg_engine: AsyncEngine, retriever: Retriever, reset_app_state: None) -> AsyncIterator[Api]:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions() as session:
        await seed_database(session, anchor=ANCHOR, reset=True)
    server = build_mcp_server(
        ToolServerDeps(
            session_factory=sessions, auth_secret=TEST_MCP_SECRET, clock=lambda: ANCHOR, retriever=retriever
        )
    )
    stt, tts = FakeSTT(), FakeTTS()
    services = Services(
        settings=api_settings(),
        sessions=sessions,
        assistant=BankingAssistant(
            AgentDeps(gateway=MCPToolGateway(server, auth_secret=TEST_MCP_SECRET), clock=lambda: ANCHOR)
        ),
        session_secret=SESSION_SECRET,
        conversations=ConversationStore(timedelta(minutes=30)),
        stt=stt,  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
    )
    app = create_app(services)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bankai.test") as client:
            yield Api(client, services, stt, tts)
    await services.aclose()


async def test_health_sees_the_postgres_database(pg_api: Api) -> None:
    response = await pg_api.client.get("/health")
    assert response.status_code == 200 and response.json()["database"] == "ok"


async def test_balances_are_exact_decimals_from_numeric_columns(pg_api: Api) -> None:
    response = await pg_api.client.get(f"/accounts/{NINO}", headers=await pg_api.login())

    assert response.status_code == 200
    expected = {a.account_number: a.balance for a in SEED.accounts if a.user_id == NINO}
    returned = {a["account_number"]: a["balance"] for a in response.json()["accounts"]}
    # Strings on the wire, and exactly the seeded NUMERIC values - no float rounding anywhere.
    assert {number: Decimal(balance) for number, balance in returned.items()} == expected
    assert all(isinstance(balance, str) for balance in returned.values())


async def test_transaction_period_filters_match_the_oracle(pg_api: Api) -> None:
    response = await pg_api.client.get(
        f"/transactions/{NINO}", params={"period": "last_30_days", "limit": 100}, headers=await pg_api.login()
    )

    body = response.json()
    assert response.status_code == 200
    assert (body["period"]["start_date"], body["period"]["end_date"]) == ("2026-08-30", "2026-09-28")
    period = resolve_period(Period.LAST_30_DAYS, now=ANCHOR)
    assert body["total_matches"] == len(user_transactions(NINO, period.start, period.end))


@pytest.mark.parametrize("requested", [GIORGI, uuid.uuid4()])
async def test_other_and_unknown_customers_look_the_same(pg_api: Api, requested: uuid.UUID) -> None:
    response = await pg_api.client.get(f"/accounts/{requested}", headers=await pg_api.login())

    assert response.status_code == 403
    assert response.json()["error"]["message"] == "You can only access your own banking information."


async def test_a_chat_turn_is_answered_and_audited_in_postgres(pg_api: Api) -> None:
    headers = await pg_api.login()
    body = await pg_api.chat(headers, "How much did I spend on groceries last month?")

    assert body["trace"]["tool"] == "analyze_spending" and "GEL" in body["response"]
    async with pg_api.services.sessions() as session:
        row = await session.scalar(
            select(ConversationLog).where(ConversationLog.request_id == body["request_id"])
        )
    assert row is not None and row.user_id == NINO
    assert (row.detected_intent, row.selected_tool) == ("spending_analysis", "analyze_spending")
    assert row.tool_arguments == body["trace"]["tool_arguments"]  # JSONB round trip


async def test_an_evaluation_run_is_stored_as_jsonb_and_read_back(pg_api: Api) -> None:
    started = await pg_api.client.post(
        "/evaluations/run", json={"categories": ["SAFETY", "GENERAL"], "limit": 6}, headers=OPERATOR
    )
    assert started.status_code == 202
    assert pg_api.services.evaluation_task is not None
    await pg_api.services.evaluation_task

    run_id = started.json()["run_id"]
    detail = (await pg_api.client.get(f"/evaluations/{run_id}", headers=OPERATOR)).json()
    assert detail["status"] == "completed" and len(detail["results"]) == detail["total_cases"] == 6
    for metric in detail["summary"]["metrics"]:
        if metric["denominator"]:
            assert metric["value"] == pytest.approx(
                100 * metric["numerator"] / metric["denominator"], abs=0.05
            )
    async with pg_api.services.sessions() as session:
        stored = (
            await session.scalars(select(EvaluationResult).where(EvaluationResult.run_id == run_id))
        ).all()
    assert {r.case_id for r in stored} == {r["case_id"] for r in detail["results"]}


async def test_concurrent_customers_are_isolated_and_fully_audited(pg_api: Api) -> None:
    """The concurrency check of tests/api/test_concurrency.py, on the production database engine."""
    headers = {number: await pg_api.login(number) for number in CUSTOMERS}
    jobs = [(number, question) for _, question, number in itertools.product(range(3), QUESTIONS, CUSTOMERS)]

    replies: list[Any] = await asyncio.gather(*(pg_api.chat(headers[n], q) for n, q in jobs))

    for (number, _), reply in zip(jobs, replies, strict=True):
        others = [m for other, uid in CUSTOMERS.items() if other != number for m in markers(uid)]
        assert not [m for m in others if m in reply["response"]]
    async with pg_api.services.sessions() as session:
        rows = (await session.scalars(select(ConversationLog))).all()
    by_request = {row.request_id: row.user_id for row in rows}
    assert [by_request.get(r["request_id"]) for r in replies] == [CUSTOMERS[n] for n, _ in jobs]
