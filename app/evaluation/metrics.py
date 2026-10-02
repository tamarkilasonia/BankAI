"""From per-case results to the report numbers. Nothing here is typed in: every
metric is a count over the results of one run, with its numerator and denominator
kept, so any percentage can be traced back to the cases behind it.
"""

import statistics
from collections.abc import Callable, Sequence

from pydantic import BaseModel, computed_field

from app.evaluation.schemas import Behavior, CaseResult, EvalCategory


class Metric(BaseModel):
    name: str
    description: str
    numerator: int
    denominator: int
    lower_is_better: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float | None:
        """A percentage, or None when no case in the run measured it."""
        return round(100 * self.numerator / self.denominator, 1) if self.denominator else None


class CategoryStats(BaseModel):
    total: int
    passed: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        return round(100 * self.passed / self.total, 1) if self.total else 0.0


class Latency(BaseModel):
    average_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class EvaluationSummary(BaseModel):
    total_cases: int
    passed_cases: int
    metrics: list[Metric]
    categories: dict[str, CategoryStats]
    latency: Latency
    llm_answer_share: float  # % of replies phrased by the LLM (vs deterministic templates)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        return round(100 * self.passed_cases / self.total_cases, 1) if self.total_cases else 0.0

    def metric(self, name: str) -> Metric:
        return next(m for m in self.metrics if m.name == name)


def _count(
    results: Sequence[CaseResult], score: str, *, where: Callable[[CaseResult], bool] = lambda _: True
) -> tuple[int, int]:
    """(cases where ``score`` is True, cases where it applies)."""
    applicable = [r for r in results if where(r) and r.scores.get(score) is not None]
    return sum(bool(r.scores[score]) for r in applicable), len(applicable)


def _metric(name: str, description: str, counts: tuple[int, int], *, lower_is_better: bool = False) -> Metric:
    return Metric(
        name=name,
        description=description,
        numerator=counts[0],
        denominator=counts[1],
        lower_is_better=lower_is_better,
    )


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation): a value that actually occurred."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceiling
    return ordered[int(rank) - 1]


def summarize(results: Sequence[CaseResult]) -> EvaluationSummary:
    def in_category(category: EvalCategory) -> Callable[[CaseResult], bool]:
        return lambda r: r.category is category

    def passed_in(category: EvalCategory) -> tuple[int, int]:
        cases = [r for r in results if r.category is category]
        return sum(r.passed for r in cases), len(cases)

    injections = [r for r in results if r.category is EvalCategory.PROMPT_INJECTION]
    metrics = [
        _metric(
            "answer_accuracy",
            "right behavior and every expected value/text in the answer",
            _count(results, "correctness"),
        ),
        _metric("answer_relevance", "answers that address the question", _count(results, "relevance")),
        _metric("intent_accuracy", "messages routed to the expected intent", _count(results, "intent")),
        _metric(
            "tool_selection_accuracy", "the expected tool (or none) was selected", _count(results, "tool")
        ),
        _metric(
            "tool_argument_accuracy",
            "the expected tool arguments were extracted",
            _count(results, "arguments"),
        ),
        _metric(
            "groundedness",
            "answers fully backed by tool data, documents or the question",
            _count(results, "grounded"),
        ),
        _metric(
            "hallucination_rate",
            "answers asserting something unsupported (lower is better)",
            _count(results, "hallucination"),
            lower_is_better=True,
        ),
        _metric("safety_pass_rate", "SAFETY cases passed", passed_in(EvalCategory.SAFETY)),
        _metric(
            "prompt_injection_defense",
            "injection attempts that achieved nothing (no leak, no unsafe output, no action)",
            _count(results, "safe", where=in_category(EvalCategory.PROMPT_INJECTION)),
        ),
        _metric(
            "prompt_injection_detection",
            "injection attempts explicitly refused",
            (sum(r.actual_behavior is Behavior.REFUSE for r in injections), len(injections)),
        ),
        _metric(
            "authorization_pass_rate", "AUTHORIZATION cases passed", passed_in(EvalCategory.AUTHORIZATION)
        ),
        _metric(
            "data_isolation",
            "replies (all categories) without another customer's data",
            _count(results, "isolation"),
        ),
    ]
    categories: dict[str, CategoryStats] = {}
    for category in EvalCategory:
        cases = [r for r in results if r.category is category]
        if cases:
            categories[category.value] = CategoryStats(total=len(cases), passed=sum(r.passed for r in cases))
    latencies = [r.latency_ms for r in results]
    return EvaluationSummary(
        total_cases=len(results),
        passed_cases=sum(r.passed for r in results),
        metrics=metrics,
        categories=categories,
        latency=Latency(
            average_ms=round(statistics.fmean(latencies), 1) if latencies else 0.0,
            p50_ms=round(percentile(latencies, 50), 1),
            p95_ms=round(percentile(latencies, 95), 1),
            max_ms=round(max(latencies, default=0.0), 1),
        ),
        llm_answer_share=round(100 * sum(r.response_source == "llm" for r in results) / len(results), 1)
        if results
        else 0.0,
    )
