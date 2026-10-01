"""Alembic migrations against real PostgreSQL."""

import asyncio

import pytest
from alembic import command
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.integration.conftest import alembic_config

pytestmark = pytest.mark.postgres

APP_TABLES = {
    "users",
    "accounts",
    "transactions",
    "banking_products",
    "conversation_logs",
    "evaluation_runs",
    "evaluation_results",
}


def _table_names(conn: Connection) -> set[str]:
    return set(inspect(conn).get_table_names())


def _check_constraint_names(conn: Connection) -> set[str]:
    inspector = inspect(conn)
    return {str(check["name"]) for table in APP_TABLES for check in inspector.get_check_constraints(table)}


def test_models_and_migrations_are_in_sync(database_url: str) -> None:
    """Fails if someone changed a model but forgot to write a migration."""
    command.check(alembic_config(database_url))


async def test_migration_creates_all_tables(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        tables = await conn.run_sync(_table_names)

    assert tables >= APP_TABLES | {"alembic_version"}


async def test_migration_creates_integrity_constraints(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        names = await conn.run_sync(_check_constraint_names)

    assert {
        "ck_accounts_balance_non_negative",
        "ck_transactions_amount_positive",
        "ck_transactions_transactioncategory",
        "ck_accounts_currency",
    } <= names


async def test_downgrade_to_base_and_upgrade_again(database_url: str, pg_engine: AsyncEngine) -> None:
    config = alembic_config(database_url)
    # Alembic's env.py runs its own event loop, so it must run in a worker thread.
    await asyncio.to_thread(command.downgrade, config, "base")
    async with pg_engine.connect() as conn:
        assert not (await conn.run_sync(_table_names)) & APP_TABLES

    await asyncio.to_thread(command.upgrade, config, "head")
    async with pg_engine.connect() as conn:
        assert await conn.run_sync(_table_names) >= APP_TABLES
