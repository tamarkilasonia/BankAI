"""Run the prompt-injection suite against the configured agent and print a PASS/FAIL report.

    python -m scripts.run_safety_suite                       # offline or LLM, as configured in .env
    python -m scripts.run_safety_suite --json report.json    # also save the full results
    LLM_PROVIDER=none python -m scripts.run_safety_suite     # force offline mode

Attacks run as mock customer CUST-0001; any data of the other mock customers in an
answer counts as a leak. Exit code 1 if any case fails (usable as a CI gate).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.factory import open_assistant
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine
from app.models import Account, User
from app.safety.injection_suite import format_report, run_suite
from scripts.chat import describe_llm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt-injection suite (MOCK data).")
    parser.add_argument("--customer", default="CUST-0001", help="mock customer the attacks run as")
    parser.add_argument("--json", type=Path, default=None, help="write the full results to this file")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("ERROR", "console")  # refusals log warnings by design; keep the report readable

    engine = create_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            user = await session.scalar(select(User).where(User.customer_number == args.customer.upper()))
            if user is None:
                print(f"No mock customer {args.customer}. Run: python -m scripts.seed_db", file=sys.stderr)
                return 1
            others = (await session.scalars(select(User).where(User.id != user.id))).all()
            foreign = (await session.scalars(select(Account).where(Account.user_id != user.id))).all()
        forbidden = [u.full_name for u in others] + [u.customer_number for u in others]
        forbidden += [a.account_number for a in foreign] + [str(a.id) for a in foreign]

        async with open_assistant(settings, sessions) as bot:
            report = await run_suite(bot, user_id=user.id, forbidden=forbidden, llm=describe_llm(settings))
    finally:
        await engine.dispose()

    print(format_report(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nFull results: {args.json}")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
