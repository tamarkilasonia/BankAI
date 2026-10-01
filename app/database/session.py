"""Async database engine and session management.

The engine is created lazily (on first use) instead of at import time, so that
importing application modules never opens a database connection. This keeps
unit tests fast and lets tests point the app at a different database.
"""

import asyncio
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

from app.core.config import get_settings


def create_engine(database_url: str, *, echo: bool = False, **kwargs: Any) -> AsyncEngine:
    """Create an async engine for the given URL.

    SQLite does not enforce foreign keys unless asked to, so we switch them on
    for every connection; otherwise unit tests could pass with data that
    PostgreSQL would reject.
    """
    engine = create_async_engine(database_url, echo=echo, pool_pre_ping=True, **kwargs)

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_conn: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache
def get_engine() -> AsyncEngine:
    """The process-wide engine, configured from settings."""
    settings = get_settings()
    return create_engine(settings.database_url, echo=settings.database_echo)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: objects stay readable after commit, which is what
    # we want when returning them from API handlers.
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    async with get_sessionmaker()() as session:
        yield session


async def check_database(engine: AsyncEngine, timeout_seconds: float = 3.0) -> bool:
    """Return True if the database answers a trivial query within the timeout.

    The timeout matters: asyncpg waits up to 60s for an unreachable host by
    default, which would make a health check hang instead of failing fast.
    """
    try:
        async with asyncio.timeout(timeout_seconds), engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, TimeoutError):
        return False
    return True
