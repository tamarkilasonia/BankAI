"""Isolation under concurrency: many customers at once, and nobody sees anybody else's data.

One app, one assistant, one LangGraph graph and one checkpointer serve every request.
A bug in per-request state - a shared thread id, a module-level variable, a mixed-up
conversation - would not show up one request at a time. Here it would show up as one
customer's numbers in another customer's answer or audit row.

These tests need a database with real separate connections: the in-memory SQLite of the
other unit tests shares ONE connection, and concurrent commits on it fail ("cannot commit
transaction - SQL statements in progress") - a test-setup artifact that made audit rows
go missing here. ``tests/integration/test_api_postgres.py`` repeats the check on PostgreSQL.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

import app.models  # noqa: F401  - registers all tables on Base.metadata
from app.database.base import Base
from app.database.session import create_engine
from app.models import ConversationLog
from tests.api.conftest import Api
from tests.constants import EMMA, GIORGI, NINO
from tests.oracle import SEED

CUSTOMERS = {"CUST-0001": NINO, "CUST-0002": GIORGI, "CUST-0010": EMMA}
QUESTIONS = ["What's my current balance?", "Show my account details.", "How much did I spend last month?"]


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """Overrides the shared in-memory engine: a database file, one connection per session."""
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'bank.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def markers(user_id: object) -> list[str]:
    """Strings that identify a customer's accounts in an answer."""
    accounts = [a for a in SEED.accounts if a.user_id == user_id]
    return [f"{a.balance:,.2f}" for a in accounts] + [a.account_number for a in accounts]


async def test_concurrent_customers_only_ever_see_their_own_data(api: Api) -> None:
    headers = {number: await api.login(number) for number in CUSTOMERS}
    # Interleaved: customer 1, 2, 3, 1, 2, 3 ... all in flight at the same time.
    jobs = [(number, question) for _, question, number in itertools.product(range(4), QUESTIONS, CUSTOMERS)]

    replies: list[Any] = await asyncio.gather(*(api.chat(headers[n], q) for n, q in jobs))

    for (number, question), reply in zip(jobs, replies, strict=True):
        others = [m for other, uid in CUSTOMERS.items() if other != number for m in markers(uid)]
        leaked = [m for m in others if m in reply["response"]]
        assert not leaked, f"{number} asked {question!r} and saw {leaked}"
        if question == QUESTIONS[0]:  # not vacuous: the balance answer does show the customer's own data
            assert any(m in reply["response"] for m in markers(CUSTOMERS[number]))
    assert len({r["request_id"] for r in replies}) == len(jobs)

    async with api.services.sessions() as session:
        rows = (await session.scalars(select(ConversationLog))).all()
    by_request = {row.request_id: row.user_id for row in rows}
    # every turn is audited, and every audit row names the right customer
    assert [by_request.get(r["request_id"]) for r in replies] == [CUSTOMERS[n] for n, _ in jobs]


async def test_concurrent_confirmations_stay_with_their_customer(api: Api) -> None:
    nino, giorgi = await api.login("CUST-0001"), await api.login("CUST-0002")
    nino_asked, giorgi_asked = await asyncio.gather(
        api.chat(nino, "Transfer 100 GEL to my friend"), api.chat(giorgi, "Transfer 250 GEL to my landlord")
    )

    # Giorgi answers "yes" in Nino's conversation at the same moment Nino does.
    nino_yes, hijack = await asyncio.gather(
        api.chat(nino, "yes", nino_asked["conversation_id"]),
        api.chat(giorgi, "yes", nino_asked["conversation_id"]),
    )
    giorgi_yes = await api.chat(giorgi, "yes", giorgi_asked["conversation_id"])

    assert Decimal(nino_yes["action_request"]["amount"]) == 100
    assert hijack["action_request"] is None and hijack["conversation_id"] != nino_asked["conversation_id"]
    assert Decimal(giorgi_yes["action_request"]["amount"]) == 250
    assert {nino_yes["action_request"]["status"], giorgi_yes["action_request"]["status"]} == {
        "recorded_mock_not_executed"
    }
