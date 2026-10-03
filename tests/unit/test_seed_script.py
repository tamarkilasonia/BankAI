"""The ``python -m scripts.seed_db`` command-line entry point."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

import app.models  # noqa: F401
from app.database.base import Base
from app.database.session import create_engine
from app.models import User
from scripts.seed_db import _anchor, main, parse_args


@pytest.fixture
async def sqlite_file_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> str:
    """A file-based SQLite database with the schema, exposed via DATABASE_URL."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'seed.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    return url


async def user_count(url: str) -> int:
    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            return (await conn.execute(select(func.count()).select_from(User))).scalar_one()
    finally:
        await engine.dispose()


async def test_seeds_an_empty_database(sqlite_file_url: str) -> None:
    assert await main(["--anchor", "2026-09-28"]) == 0
    assert await user_count(sqlite_file_url) == 10


async def test_refuses_to_seed_twice_and_reset_succeeds(sqlite_file_url: str) -> None:
    assert await main(["--anchor", "2026-09-28"]) == 0

    assert await main(["--anchor", "2026-09-28"]) == 1  # already seeded
    assert await main(["--reset", "--seed", "7"]) == 0
    assert await user_count(sqlite_file_url) == 10


async def test_if_empty_seeds_once_and_then_does_nothing(sqlite_file_url: str) -> None:
    """The Docker init job runs on every `docker compose up`: it must never duplicate data."""
    assert await main(["--if-empty", "--anchor", "2026-09-28"]) == 0
    assert await main(["--if-empty"]) == 0
    assert await user_count(sqlite_file_url) == 10


def test_if_empty_and_reset_exclude_each_other() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--if-empty", "--reset"])


def test_anchor_means_end_of_day_in_bank_time() -> None:
    # 23:59:59 in Tbilisi (UTC+4) is 19:59:59 UTC.
    assert _anchor("2026-09-28") == datetime(2026, 9, 28, 19, 59, 59, tzinfo=UTC)


def test_invalid_anchor_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--anchor", "28/09/2026"])


def test_defaults() -> None:
    args = parse_args([])
    assert (args.reset, args.anchor, args.seed) == (False, None, None)
