"""The ingest and calibration command-line scripts."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

import app.models  # noqa: F401
from app.database.base import Base
from app.database.session import create_engine
from app.models import KnowledgeChunk
from scripts import calibrate_rag, ingest_knowledge_base
from tests.conftest import PROJECT_ROOT


@pytest.fixture
async def sqlite_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> str:
    url = f"sqlite+aiosqlite:///{tmp_path / 'kb.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(PROJECT_ROOT / "knowledge_base"))
    return url


async def stored_chunks(url: str) -> int:
    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            return (await conn.execute(select(func.count()).select_from(KnowledgeChunk))).scalar_one()
    finally:
        await engine.dispose()


async def test_ingest_indexes_once_and_then_skips(
    sqlite_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await ingest_knowledge_base.main([]) == 0
    first = capsys.readouterr().out
    assert await ingest_knowledge_base.main([]) == 0
    second = capsys.readouterr().out
    assert await ingest_knowledge_base.main(["--force"]) == 0

    assert "chunks from 11 documents indexed (embedder: hashing-v1-768)" in first
    assert "already up to date" in second
    assert await stored_chunks(sqlite_url) > 40


async def test_ingest_with_memory_store_has_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], reset_app_state: None
) -> None:
    monkeypatch.setenv("VECTOR_STORE", "memory")
    assert await ingest_knowledge_base.main([]) == 0
    assert "nothing to ingest" in capsys.readouterr().out


async def test_ingest_reports_embedding_failure(
    sqlite_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")

    assert await ingest_knowledge_base.main([]) == 1
    assert "Embedding failed" in capsys.readouterr().err


@pytest.mark.usefixtures("reset_app_state")
async def test_calibration_script_reports_a_positive_margin(capsys: pytest.CaptureFixture[str]) -> None:
    assert await calibrate_rag.main() == 0

    out = capsys.readouterr().out
    assert "right document in top 2: 16/16" in out
    assert "suggested RAG_MIN_SCORE: 0.255" in out
