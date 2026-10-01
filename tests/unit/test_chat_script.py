"""The terminal chat (``python -m scripts.chat``)."""

import builtins
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

import app.models  # noqa: F401
from app.core.config import Settings
from app.database.base import Base
from app.database.seed import seed_database
from app.database.session import create_engine
from scripts.chat import describe_llm, main
from tests.constants import ANCHOR


@pytest.fixture
async def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> str:
    url = f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        await seed_database(session, anchor=ANCHOR)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def feed(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    replies: Iterator[str] = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


@pytest.mark.usefixtures("seeded_db")
async def test_one_shot_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert await main(["--message", "What's my balance?", "--debug"]) == 0

    out = capsys.readouterr().out
    assert "bankai> Here are your current balances:" in out
    assert "intent=account_balance via rules | tool=get_account_balance" in out


@pytest.mark.usefixtures("seeded_db")
async def test_interactive_session_keeps_follow_up_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The CLI runs on the real clock, so dates are not asserted - only that turn 2 is a follow-up.
    feed(monkeypatch, "how much did I spend on restaurants this month", "", "and last month?", "exit")

    assert await main(["--customer", "cust-0003", "--debug"]) == 0

    out = capsys.readouterr().out
    assert "Signed in as Ana Lomidze (CUST-0003)" in out
    assert out.count("bankai> ") == 2
    assert "intent=spending_analysis via follow_up" in out
    assert "'period': 'last_month'" in out


@pytest.mark.usefixtures("seeded_db")
async def test_end_of_input_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    feed(monkeypatch)
    assert await main([]) == 0


@pytest.mark.usefixtures("seeded_db")
async def test_unknown_customer(capsys: pytest.CaptureFixture[str]) -> None:
    assert await main(["--customer", "CUST-9999", "--message", "hi"]) == 1
    assert "No mock customer CUST-9999" in capsys.readouterr().err


@pytest.mark.usefixtures("seeded_db")
async def test_remote_mcp_server_unavailable_is_reported_in_the_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        await main(["--mcp-url", "http://127.0.0.1:9/mcp", "--message", "What's my balance?", "--debug"]) == 0
    )
    out = capsys.readouterr().out
    assert "temporarily unavailable" in out
    assert "[errors: tool:unavailable]" in out


def test_describe_llm() -> None:
    assert describe_llm(Settings(_env_file=None, llm_provider="none")).startswith("none")
    ollama = Settings(
        _env_file=None, llm_provider="openai_compatible", llm_base_url="http://localhost:11434/v1"
    )
    assert describe_llm(ollama) == "qwen2.5:3b via http://localhost:11434/v1"
