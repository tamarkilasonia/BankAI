"""Integration-test fixtures: a real PostgreSQL database, migrated with Alembic.

* ``TEST_DATABASE_URL`` selects the database (default: local ``bankai_test``).
* If it is unreachable the tests are SKIPPED locally (with a clear reason),
  but FAIL when ``REQUIRE_INTEGRATION_DB=1`` - CI sets this, so a broken
  database can never silently turn into a green build.
* As a safety net the database name must end in ``_test``: these tests wipe
  the schema and must never be pointed at a development database.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.database.session import check_database, create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DATABASE_URL = "postgresql+asyncpg://bankai:bankai@localhost:5432/bankai_test"


def alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    config.attributes["configure_logger"] = False  # keep pytest's logging setup
    return config


async def _reachable(url: str) -> bool:
    engine = create_engine(url)
    try:
        return await check_database(engine)
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """URL of a reachable, freshly migrated test database."""
    url = os.getenv("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail(f"Refusing to wipe database {database!r}: integration test databases must end in _test")

    # Sync session fixture: no event loop is running yet, so asyncio.run is safe.
    if not asyncio.run(_reachable(url)):
        message = f"PostgreSQL not reachable at {make_url(url).render_as_string(hide_password=True)}"
        if os.getenv("REQUIRE_INTEGRATION_DB") == "1":
            pytest.fail(message)
        pytest.skip(message)

    config = alembic_config(url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield url


@pytest.fixture
async def pg_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_engine(database_url)
    yield engine
    await engine.dispose()


@pytest.fixture
async def pg_session(pg_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with async_sessionmaker(pg_engine, expire_on_commit=False)() as session:
        yield session
