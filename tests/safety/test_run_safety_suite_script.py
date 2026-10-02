"""``python -m scripts.run_safety_suite`` against a seeded database (offline agent)."""

import json
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

import app.models  # noqa: F401
from app.database.base import Base
from app.database.seed import seed_database
from app.database.session import create_engine
from app.safety import injection_suite
from scripts import run_safety_suite
from tests.constants import ANCHOR


@pytest.fixture
async def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'bank.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        await seed_database(session, anchor=ANCHOR)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("VECTOR_STORE", "memory")


@pytest.mark.usefixtures("seeded_db")
async def test_report_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "report.json"

    assert await run_safety_suite.main(["--json", str(target)]) == 0

    out = capsys.readouterr().out
    assert "Attacks defended:            57/57 (100.0%)" in out
    assert "False refusals (benign):     0/26 (0.0%)" in out
    assert len(json.loads(target.read_text())["results"]) == 83


@pytest.mark.usefixtures("seeded_db")
async def test_any_failure_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    real = injection_suite.judge
    monkeypatch.setattr(run_safety_suite, "run_suite", injection_suite.run_suite)
    monkeypatch.setattr(
        injection_suite,
        "judge",
        lambda case, reply, forbidden: (
            real(case, reply, forbidden) or ("forced failure" if case.id == "PI-001" else None)
        ),
    )

    assert await run_safety_suite.main([]) == 1


@pytest.mark.usefixtures("seeded_db")
async def test_unknown_customer(capsys: pytest.CaptureFixture[str]) -> None:
    assert await run_safety_suite.main(["--customer", "CUST-9999"]) == 1
    assert "No mock customer CUST-9999" in capsys.readouterr().err
