"""Fill the database with deterministic MOCK banking data.

Usage (from the project root, after ``alembic upgrade head``)::

    python -m scripts.seed_db                      # seed an empty database
    python -m scripts.seed_db --reset              # replace existing mock data
    python -m scripts.seed_db --anchor 2026-09-28  # pin "today" of the mock history
    python -m scripts.seed_db --if-empty           # seed only a fresh database (container start-up)
"""

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from typing import cast

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.time import BANK_TZ
from app.database.seed import SeedError, has_mock_data, seed_database
from app.database.session import create_engine

logger = get_logger("scripts.seed_db")


def _anchor(value: str) -> datetime:
    """``YYYY-MM-DD`` -> the end of that day in bank time, as UTC."""
    day = date.fromisoformat(value)
    return datetime.combine(day, time(23, 59, 59), tzinfo=BANK_TZ).astimezone(UTC)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the BankAI database with MOCK data.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reset", action="store_true", help="delete existing mock data first")
    mode.add_argument(
        "--if-empty", action="store_true", help="do nothing if the database already has mock data"
    )
    parser.add_argument(
        "--anchor",
        type=_anchor,
        default=None,
        help="'today' of the mock history as YYYY-MM-DD (default: now)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: SEED_RANDOM_SEED)")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    engine = create_engine(settings.database_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            if args.if_empty and await has_mock_data(session):
                logger.info("seed.skipped", extra={"reason": "the database already has mock data"})
                return 0
            summary = await seed_database(
                session,
                anchor=cast(datetime | None, args.anchor),
                seed=args.seed if args.seed is not None else settings.seed_random_seed,
                reset=args.reset,
            )
    except SeedError as exc:
        logger.error("seed.refused", extra={"reason": str(exc)})
        return 1
    finally:
        await engine.dispose()

    logger.info("seed.completed", extra={**asdict(summary), "database": settings.database_url_safe})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
