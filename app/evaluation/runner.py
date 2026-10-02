"""Running the evaluation.

The agent is evaluated against a **frozen MOCK bank**: a fresh in-memory
database seeded with a fixed anchor date, with the agent's clock pinned to that
date. Every run therefore sees the same customers, transactions and "today", so
runs are comparable - a change in a metric means the agent changed, not the data.

The oracle queries the same frozen database, so expected values always match
the data the agent was given.
"""

import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - registers every table on Base.metadata
from app.agents.assistant import AssistantReply, BankingAssistant, PendingConfirmation, PreviousTurn
from app.core.config import Settings
from app.core.logging import get_logger
from app.database.base import Base
from app.database.seed import seed_database
from app.database.session import create_engine
from app.evaluation import store
from app.evaluation.dataset import DATASET, dataset_version, load_cases
from app.evaluation.metrics import EvaluationSummary, summarize
from app.evaluation.oracle import SqlOracle
from app.evaluation.schemas import CaseResult, EvalCase
from app.evaluation.scoring import score_case
from app.mcp.client import MCPToolGateway
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.rag.factory import create_retriever

logger = get_logger(__name__)

# "Now" in the evaluation bank: Monday 28 September 2026, 22:00 bank time.
EVAL_ANCHOR = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)

Progress = Callable[[int, int, CaseResult], None]


class EvaluationReport(BaseModel):
    run_id: str | None
    llm_provider: str
    model_name: str | None
    dataset_name: str
    dataset_version: str
    started_at: datetime
    duration_s: float
    summary: EvaluationSummary
    results: list[CaseResult]

    @property
    def llm_label(self) -> str:
        return "none (offline: rules + templates)" if self.llm_provider == "none" else str(self.model_name)


@asynccontextmanager
async def frozen_bank(anchor: datetime = EVAL_ANCHOR) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A private, freshly seeded in-memory database (nothing persists, nothing is shared)."""
    engine = create_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            await seed_database(session, anchor=anchor)
        yield sessions
    finally:
        await engine.dispose()


async def ask_case(assistant: BankingAssistant, case: EvalCase, user_id: object) -> AssistantReply:
    """Play the case's earlier turns (follow-ups, confirmations), then ask the evaluated message."""
    previous: PreviousTurn | None = None
    pending: PendingConfirmation | None = None
    for message in [*case.history, case.input]:
        reply = await assistant.respond(
            user_id=user_id,  # type: ignore[arg-type]
            message=message,
            channel=case.channel,
            previous=previous,
            pending=pending,
        )
        previous = reply.as_previous_turn() or previous
        pending = reply.pending_confirmation
    return reply


async def run_cases(
    assistant: BankingAssistant,
    oracle: SqlOracle,
    cases: Sequence[EvalCase],
    *,
    progress: Progress | None = None,
) -> list[CaseResult]:
    results = []
    for index, case in enumerate(cases, start=1):
        user_id = await oracle.customer_id(case.customer)
        reply = await ask_case(assistant, case, user_id)
        expected = (
            await oracle.answer(case.expected_answer.oracle, user_id) if case.expected_answer.oracle else None
        )
        result = score_case(case, reply, oracle=expected, foreign=await oracle.foreign_markers(user_id))
        results.append(result)
        if progress is not None:
            progress(index, len(cases), result)
    return results


async def evaluate(
    settings: Settings,
    *,
    cases: Sequence[EvalCase] | None = None,
    dataset: Path = DATASET,
    results_store: async_sessionmaker[AsyncSession] | None = None,
    run_id: uuid.UUID | None = None,
    progress: Progress | None = None,
) -> EvaluationReport:
    """Run ``cases`` (default: the whole dataset) with the agent configured by ``settings``.

    ``results_store``: where to save the run (the application database); None = don't save.
    ``run_id``: a run already created with ``store.start_run`` (the API answers with its ID first).
    """
    cases = list(cases) if cases is not None else load_cases(dataset)
    model = None if settings.llm_provider == "none" else settings.llm_model
    version = dataset_version(dataset)
    started_at, started = datetime.now(UTC), time.perf_counter()
    if results_store is not None and run_id is None:
        run_id = await store.start_run(
            results_store,
            llm_provider=settings.llm_provider,
            model_name=model,
            dataset_name=dataset.stem,
            dataset_version=version,
        )
    try:
        async with frozen_bank() as sessions:
            # The frozen bank is SQLite, so documents are indexed in memory (with the configured embedder).
            retriever = create_retriever(settings.model_copy(update={"vector_store": "memory"}), sessions)
            secret = secrets.token_urlsafe(32)
            server = build_mcp_server(
                ToolServerDeps(
                    session_factory=sessions,
                    auth_secret=secret,
                    clock=lambda: EVAL_ANCHOR,
                    retriever=retriever,
                )
            )
            gateway = MCPToolGateway(
                server, auth_secret=secret, timeout_seconds=settings.tool_timeout_seconds
            )
            assistant = BankingAssistant.from_settings(
                settings, gateway=gateway, extra_secrets=[secret], clock=lambda: EVAL_ANCHOR
            )
            try:
                results = await run_cases(
                    assistant, SqlOracle(sessions, now=EVAL_ANCHOR), cases, progress=progress
                )
            finally:
                await assistant.aclose()
                await retriever.aclose()
    except Exception as exc:
        if results_store is not None and run_id is not None:
            await store.fail_run(results_store, run_id, f"{type(exc).__name__}: {exc}")
        raise

    summary = summarize(results)
    if results_store is not None and run_id is not None:
        await store.finish_run(results_store, run_id, summary, results)
    logger.info(
        "evaluation.completed",
        extra={
            "run_id": str(run_id) if run_id else None,
            "cases": len(results),
            "passed": summary.passed_cases,
        },
    )
    return EvaluationReport(
        run_id=str(run_id) if run_id else None,
        llm_provider=settings.llm_provider,
        model_name=model,
        dataset_name=dataset.stem,
        dataset_version=version,
        started_at=started_at,
        duration_s=round(time.perf_counter() - started, 1),
        summary=summary,
        results=results,
    )
