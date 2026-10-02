"""Persisting evaluation runs (``evaluation_runs``) and their per-case results (``evaluation_results``).

A run is created as ``running`` before the first case, so a crashed or still
running evaluation is visible (the API will start runs in the background).
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.time import utcnow
from app.evaluation.metrics import EvaluationSummary
from app.evaluation.schemas import CaseResult
from app.models import EvaluationResult, EvaluationRun
from app.models.enums import EvaluationRunStatus


async def start_run(
    sessions: async_sessionmaker[AsyncSession],
    *,
    llm_provider: str,
    model_name: str | None,
    dataset_name: str,
    dataset_version: str,
) -> uuid.UUID:
    async with sessions() as session:
        run = EvaluationRun(
            llm_provider=llm_provider,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            status=EvaluationRunStatus.RUNNING,
        )
        session.add(run)
        await session.commit()
        return run.id


async def finish_run(
    sessions: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    summary: EvaluationSummary,
    results: Sequence[CaseResult],
) -> None:
    async with sessions() as session:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            raise LookupError(f"evaluation run {run_id} not found")
        run.status = EvaluationRunStatus.COMPLETED
        run.total_cases = summary.total_cases
        run.passed_cases = summary.passed_cases
        run.summary = summary.model_dump(mode="json")
        run.finished_at = utcnow()
        session.add_all(
            EvaluationResult(
                run_id=run_id,
                case_id=r.case_id,
                category=r.category.value,
                input_text=r.input,
                expected_behavior=r.expected_behavior.value,
                actual_behavior=r.actual_behavior.value,
                expected_intent=r.expected_intent,
                actual_intent=r.actual_intent,
                expected_tool=r.expected_tool,
                actual_tool=r.actual_tool,
                passed=r.passed,
                scores=r.scores,
                failure_reasons=r.failure_reasons,
                response_text=r.response,
                latency_ms=r.latency_ms,
            )
            for r in results
        )
        await session.commit()


async def fail_run(sessions: async_sessionmaker[AsyncSession], run_id: uuid.UUID, error: str) -> None:
    async with sessions() as session:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            return
        run.status = EvaluationRunStatus.FAILED
        run.summary = {"error": error}
        run.finished_at = utcnow()
        await session.commit()


async def list_runs(sessions: async_sessionmaker[AsyncSession], *, limit: int = 20) -> list[EvaluationRun]:
    async with sessions() as session:
        rows = await session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(limit)
        )
        return list(rows)


async def get_run(sessions: async_sessionmaker[AsyncSession], run_id: uuid.UUID) -> EvaluationRun | None:
    """A run with its results loaded."""
    async with sessions() as session:
        return await session.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.id == run_id)
            .options(selectinload(EvaluationRun.results))
        )
