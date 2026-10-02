"""Human-readable reports: a terminal summary and a Markdown document."""

from app.evaluation.metrics import Metric
from app.evaluation.runner import EvaluationReport

TITLES = {
    "answer_accuracy": "Answer Accuracy",
    "answer_relevance": "Answer Relevance",
    "intent_accuracy": "Intent Accuracy",
    "tool_selection_accuracy": "Tool Selection Accuracy",
    "tool_argument_accuracy": "Tool Argument Accuracy",
    "groundedness": "Groundedness",
    "hallucination_rate": "Hallucination Rate",
    "safety_pass_rate": "Safety Pass Rate",
    "prompt_injection_defense": "Prompt Injection Defense",
    "prompt_injection_detection": "Prompt Injection Detection",
    "authorization_pass_rate": "Authorization Pass Rate",
    "data_isolation": "Data Isolation",
}


def _value(metric: Metric) -> str:
    return "n/a" if metric.value is None else f"{metric.value:.1f}%"


def format_text(report: EvaluationReport, *, max_failures: int = 40) -> str:
    s = report.summary
    lines = [
        f"BankAI evaluation - dataset {report.dataset_name}@{report.dataset_version}"
        f" - LLM: {report.llm_label}",
        f"Run {report.run_id or '(not saved)'} - {report.duration_s:.1f} s",
        "",
        f"Total Tests: {s.total_cases}    Passed: {s.passed_cases}"
        f"    Failed: {s.total_cases - s.passed_cases}    ({s.pass_rate:.1f}%)",
        "",
    ]
    for metric in s.metrics:
        lines.append(
            f"{TITLES[metric.name]:<28}{_value(metric):>7}   ({metric.numerator}/{metric.denominator})"
        )
    lines += [
        f"{'Average Latency':<28}{s.latency.average_ms / 1000:>6.2f}s   "
        f"(p50 {s.latency.p50_ms / 1000:.2f}s, p95 {s.latency.p95_ms / 1000:.2f}s)",
        f"{'LLM-written answers':<28}{s.llm_answer_share:>6.1f}%",
        "",
        "By category:",
    ]
    for name, stats in s.categories.items():
        lines.append(f"  {name:<18}{stats.passed:>4}/{stats.total:<4}{stats.pass_rate:>7.1f}%")
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines += ["", f"Failed cases ({len(failed)}):"]
        for r in failed[:max_failures]:
            lines.append(f"  [{r.category.value}] {r.case_id}  {r.input!r}")
            lines.append(
                f"      expected: {r.expected_behavior.value} / {r.expected_tool}    "
                f"actual: {r.actual_behavior.value} / {r.actual_tool}"
            )
            lines += [f"      - {reason}" for reason in r.failure_reasons[:4]]
        if len(failed) > max_failures:
            lines.append(f"  ... and {len(failed) - max_failures} more (see the saved run or --json)")
    return "\n".join(lines)


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(report: EvaluationReport) -> str:
    s = report.summary
    lines = [
        "# BankAI evaluation report",
        "",
        f"* Dataset: `{report.dataset_name}` version `{report.dataset_version}` ({s.total_cases} cases)",
        f"* LLM: {report.llm_label}",
        f"* Run: `{report.run_id or 'not saved'}`, {report.started_at:%Y-%m-%d %H:%M} UTC, "
        f"{report.duration_s:.0f} s",
        "* All numbers are computed from this run; data is the frozen MOCK bank (anchor 2026-09-28).",
        "",
        f"**{s.passed_cases}/{s.total_cases} cases passed ({s.pass_rate:.1f}%).**",
        "",
        "| Metric | Value | Cases | Meaning |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {TITLES[m.name]} | {_value(m)} | {m.numerator}/{m.denominator} | {m.description} |"
        for m in s.metrics
    ]
    lines += [
        f"| Average latency | {s.latency.average_ms / 1000:.2f} s | | p50 {s.latency.p50_ms / 1000:.2f} s, "
        f"p95 {s.latency.p95_ms / 1000:.2f} s |",
        f"| LLM-written answers | {s.llm_answer_share:.1f}% | | the rest are deterministic templates |",
        "",
        "## By category",
        "",
        "| Category | Passed | Pass rate |",
        "|---|---|---|",
    ]
    lines += [
        f"| {name} | {st.passed}/{st.total} | {st.pass_rate:.1f}% |" for name, st in s.categories.items()
    ]
    failed = [r for r in report.results if not r.passed]
    lines += ["", f"## Failed cases ({len(failed)})", ""]
    if failed:
        lines += ["| Case | Input | Expected | Actual | Why |", "|---|---|---|---|---|"]
        lines += [
            f"| {r.case_id} | {_md(r.input)} | {r.expected_behavior.value} / {r.expected_tool} | "
            f"{r.actual_behavior.value} / {r.actual_tool} | {_md('; '.join(r.failure_reasons[:3]))} |"
            for r in failed
        ]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"
