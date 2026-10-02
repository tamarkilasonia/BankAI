"""Run the evaluation suite and print the report.

    python -m scripts.run_evaluation                          # whole dataset, agent as configured in .env
    python -m scripts.run_evaluation --category SPENDING      # one category (repeatable)
    python -m scripts.run_evaluation --tag holdout            # cases with a tag (repeatable)
    python -m scripts.run_evaluation --report docs/EVALUATION.md --json results.json
    LLM_PROVIDER=none python -m scripts.run_evaluation        # offline baseline (rules + templates)

The agent runs against a frozen MOCK bank (in-memory, fixed date), so runs are
comparable. The run and every case result are saved to the application
database (``evaluation_runs`` / ``evaluation_results``) unless ``--no-save``.

Exit code 1 if any answer contained another customer's data - for a bank that is
a hard failure, whatever the other metrics say.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine
from app.evaluation.dataset import load_cases, select_cases
from app.evaluation.report import format_text, to_markdown
from app.evaluation.runner import evaluate
from app.evaluation.schemas import CaseResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BankAI evaluation (MOCK data).")
    parser.add_argument("--category", action="append", default=[], help="only this category (repeatable)")
    parser.add_argument("--tag", action="append", default=[], help="only cases with this tag (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="at most this many cases")
    parser.add_argument("--no-save", action="store_true", help="do not store the run in the database")
    parser.add_argument("--report", type=Path, default=None, help="also write a Markdown report here")
    parser.add_argument("--json", type=Path, default=None, help="also write the full results as JSON here")
    parser.add_argument("--quiet", action="store_true", help="no per-case progress")
    return parser.parse_args(argv)


def show_progress(done: int, total: int, result: CaseResult) -> None:
    mark = "." if result.passed else "F"
    print(mark, end="\n" if done % 60 == 0 or done == total else "", flush=True)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("ERROR", "console")  # refusals and fallbacks log warnings by design

    cases = select_cases(load_cases(), categories=args.category, tags=args.tag, limit=args.limit)
    if not cases:
        print("No cases match the filters.", file=sys.stderr)
        return 1

    engine = None if args.no_save else create_engine(settings.database_url)
    try:
        report = await evaluate(
            settings,
            cases=cases,
            results_store=async_sessionmaker(engine, expire_on_commit=False) if engine else None,
            progress=None if args.quiet else show_progress,
        )
    finally:
        if engine is not None:
            await engine.dispose()

    print(format_text(report))
    if args.report:
        args.report.write_text(to_markdown(report), encoding="utf-8")
        print(f"\nMarkdown report: {args.report}")
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Full results: {args.json}")
    return 0 if report.summary.metric("data_isolation").numerator == report.summary.total_cases else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
