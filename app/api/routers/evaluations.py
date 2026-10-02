"""Evaluation endpoints (operators only: X-Operator-Key).

* ``GET  /evaluations``          - recent runs with their metrics (the dashboard's list).
* ``GET  /evaluations/{run_id}`` - one run with every case result (the dashboard's details).
* ``POST /evaluations/run``      - start a run in the background; answers 202 with its ID at once.
  One run at a time per server (a run with an LLM takes minutes).
"""

import asyncio
import uuid

from fastapi import APIRouter, Query, status

from app.api.dependencies import OperatorDep, ServicesDep
from app.api.errors import ApiError
from app.api.schemas import (
    ErrorResponse,
    EvaluationResultOut,
    EvaluationRunDetail,
    EvaluationRunOut,
    EvaluationRunRequest,
    EvaluationRunStarted,
)
from app.core.logging import get_logger
from app.evaluation import store
from app.evaluation.dataset import DATASET, dataset_version, load_cases, select_cases
from app.evaluation.runner import evaluate
from app.models import EvaluationRun

logger = get_logger(__name__)
router = APIRouter(prefix="/evaluations", tags=["evaluations"], dependencies=[OperatorDep])
ERRORS = {code: {"model": ErrorResponse} for code in (403, 404, 409, 422, 503)}


def _run_out(run: EvaluationRun) -> EvaluationRunOut:
    return EvaluationRunOut(
        id=run.id,
        status=run.status.value,
        llm_provider=run.llm_provider,
        model_name=run.model_name,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        total_cases=run.total_cases,
        passed_cases=run.passed_cases,
        started_at=run.started_at,
        finished_at=run.finished_at,
        summary=run.summary,
    )


@router.get("", responses=ERRORS)  # type: ignore[arg-type]
async def list_runs(
    services: ServicesDep, limit: int = Query(default=20, ge=1, le=100)
) -> list[EvaluationRunOut]:
    """Recent evaluation runs, newest first."""
    return [_run_out(run) for run in await store.list_runs(services.sessions, limit=limit)]


@router.get("/{run_id}", responses=ERRORS)  # type: ignore[arg-type]
async def get_run(run_id: uuid.UUID, services: ServicesDep) -> EvaluationRunDetail:
    """One run with every case: input, expected and actual behavior, scores, why it failed."""
    run = await store.get_run(services.sessions, run_id)
    if run is None:
        raise ApiError(404, "not_found", "No evaluation run with this ID.")
    results = sorted(run.results, key=lambda r: (r.passed, r.case_id))  # failures first
    return EvaluationRunDetail(
        **_run_out(run).model_dump(),
        results=[EvaluationResultOut.model_validate(r, from_attributes=True) for r in results],
    )


@router.post("/run", status_code=status.HTTP_202_ACCEPTED, responses=ERRORS)  # type: ignore[arg-type]
async def start_run(body: EvaluationRunRequest, services: ServicesDep) -> EvaluationRunStarted:
    """Start an evaluation of the agent as configured on this server. Poll ``GET /evaluations/{id}``."""
    if services.evaluation_task is not None and not services.evaluation_task.done():
        raise ApiError(409, "conflict", "An evaluation is already running.")
    try:
        cases = select_cases(load_cases(), categories=body.categories, tags=body.tags, limit=body.limit)
    except ValueError as exc:
        raise ApiError(422, "invalid_request", f"Unknown category: {exc}") from None
    if not cases:
        raise ApiError(422, "invalid_request", "No evaluation cases match the filters.")
    settings = services.settings
    run_id = await store.start_run(
        services.sessions,
        llm_provider=settings.llm_provider,
        model_name=None if settings.llm_provider == "none" else settings.llm_model,
        dataset_name=DATASET.stem,
        dataset_version=dataset_version(),
    )
    task = asyncio.create_task(
        evaluate(settings, cases=cases, results_store=services.sessions, run_id=run_id)
    )
    task.add_done_callback(_report_failure)
    services.evaluation_task = task
    return EvaluationRunStarted(run_id=run_id, status="running", cases=len(cases))


def _report_failure(task: "asyncio.Task[object]") -> None:
    """The runner already marked the run as failed; make sure the error is logged, not swallowed."""
    if not task.cancelled() and task.exception() is not None:
        logger.error("evaluation.background_failed", extra={"error": type(task.exception()).__name__})
