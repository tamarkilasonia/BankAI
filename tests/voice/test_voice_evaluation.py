"""The voice round-trip evaluation (WER math, decisions, report) and the voice scripts."""

from pathlib import Path
from typing import Any

import pytest

from app.evaluation import voice as voice_eval
from app.evaluation.voice import (
    character_error_rate,
    edit_distance,
    english_sentences,
    normalize,
    run_round_trips,
    to_markdown,
    word_error_rate,
)
from app.voice.errors import SpeechUnavailableError
from app.voice.stt import Transcription
from scripts import download_voice_models, evaluate_voice
from tests.fakes import FakeSTT, FakeTTS


def heard(text: str, language: str = "en", confidence: float = 0.8) -> Transcription:
    return Transcription(text, language, 0.9, confidence, 1.0, "fake", 10.0)


def test_normalization_ignores_case_punctuation_apostrophes_and_small_number_words() -> None:
    assert normalize("What's my balance? Two years, 10,000 GEL!") == [
        "whats",
        "my",
        "balance",
        "2",
        "years",
        "10000",
        "gel",
    ]


def test_error_rates() -> None:
    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert word_error_rate("show me my balance in lari", "Show me my balance and lorry.") == pytest.approx(
        2 / 6
    )
    assert word_error_rate("", "") == 0.0
    assert character_error_rate("abc", "abd") == pytest.approx(1 / 3)


def test_english_sample_comes_from_the_dataset() -> None:
    sample = english_sentences()
    assert 30 <= len(sample) <= 60 and all(s.isascii() for s in sample)
    assert sample == english_sentences()  # deterministic


async def test_round_trip_counts_accepts_and_rejects() -> None:
    stt = FakeSTT(
        heard("What's my balance?"),  # perfect
        heard("What's my account's desk?", confidence=0.9),  # wrong but confident: a false accept
        heard("What's my balance today?", confidence=0.3),  # nearly right but unsure: a false reject
        heard("Ramdeni tanxa", language="fa", confidence=0.7),  # Georgian, garbled: not acted on
    )

    report = await run_round_trips(
        FakeTTS(),
        stt,  # type: ignore[arg-type]
        min_confidence=0.5,
        english=["What's my balance?", "What's my account status?", "What's my balance today?"],
        georgian=["რამდენი თანხა მაქვს?"],
        voices={"en": "fake-en", "ka": "fake-ka"},
    )

    en, ka = report.languages["en"], report.languages["ka"]
    assert (en.samples, en.exact, en.acted_on, en.false_accepts, en.false_rejects) == (3, 2, 2, 1, 1)
    assert (ka.acted_on, ka.right_script, ka.false_accepts) == (0, 0, 0)
    assert report.threshold_window == (0.9, 0.3)
    markdown = to_markdown(report)
    assert "| Acted on (passed to the agent) | 2/3 | 0/1 |" in markdown
    assert "| ka | რამდენი თანხა მაქვს? | Ramdeni tanxa |" in markdown
    assert "Highest confidence of a bad transcript: 0.900" in markdown


# --- scripts -------------------------------------------------------------------------------------------


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> None:
    """The scripts' real model classes replaced by fakes that hear what they are told to say."""

    class EchoSTT(FakeSTT):
        async def transcribe(self, audio: bytes, *, language: str | None = None) -> Any:
            return heard("echo")

    monkeypatch.setattr(evaluate_voice, "PiperTTS", lambda voices, voices_dir: FakeTTS())
    monkeypatch.setattr(evaluate_voice, "FasterWhisperSTT", lambda model, model_dir, prompt: EchoSTT())
    monkeypatch.setattr(voice_eval, "english_sentences", lambda: ["echo"])


@pytest.mark.usefixtures("fake_models")
async def test_evaluate_voice_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "voice.md"
    assert await evaluate_voice.main(["--output", str(target)]) == 0
    assert target.read_text().startswith("# Voice round trip")
    assert "en: WER 0.0%" in capsys.readouterr().out


async def test_evaluate_voice_reports_missing_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], reset_app_state: None
) -> None:
    monkeypatch.setattr(
        evaluate_voice, "PiperTTS", lambda voices, voices_dir: FakeTTS(SpeechUnavailableError("no voice"))
    )
    monkeypatch.setattr(evaluate_voice, "FasterWhisperSTT", lambda model, model_dir, prompt: FakeSTT())
    assert await evaluate_voice.main(["--output", str(tmp_path / "x.md")]) == 1
    assert "voice_unavailable: no voice" in capsys.readouterr().err


def test_download_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reset_app_state: None) -> None:
    pytest.importorskip("faster_whisper")
    pytest.importorskip("piper")
    import faster_whisper
    import piper.download_voices

    calls: list[str] = []
    monkeypatch.setenv("STT_MODEL_DIR", str(tmp_path / "w"))
    monkeypatch.setenv("TTS_VOICES_DIR", str(tmp_path / "p"))
    monkeypatch.setattr(
        faster_whisper, "download_model", lambda size, cache_dir: calls.append(size) or "path"
    )
    monkeypatch.setattr(piper.download_voices, "download_voice", lambda voice, directory: calls.append(voice))

    assert download_voice_models.main(["--whisper", "base"]) == 0
    assert calls == ["base", "en_US-lessac-medium", "ka_GE-natia-medium"]
    assert (tmp_path / "w").is_dir() and (tmp_path / "p").is_dir()


def test_download_script_without_the_packages(monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert download_voice_models.main([]) == 1


# --- python -m scripts.voice_turn -------------------------------------------------------------------


@pytest.fixture
async def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_app_state: None) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.models  # noqa: F401
    from app.database.base import Base
    from app.database.seed import seed_database
    from app.database.session import create_engine
    from tests.constants import ANCHOR

    url = f"sqlite+aiosqlite:///{tmp_path / 'bank.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        await seed_database(session, anchor=ANCHOR)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("VECTOR_STORE", "memory")


def use_voice(monkeypatch: pytest.MonkeyPatch, stt: FakeSTT | None, tts: FakeTTS | None) -> None:
    from scripts import voice_turn

    monkeypatch.setattr(voice_turn, "create_stt", lambda settings: stt)
    monkeypatch.setattr(voice_turn, "create_tts", lambda settings: tts)


@pytest.mark.usefixtures("seeded_db")
async def test_voice_turn_answers_and_saves_speech(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import voice_turn
    from tests.fakes import wav_bytes

    question, answer = tmp_path / "q.wav", tmp_path / "a.wav"
    question.write_bytes(wav_bytes())
    use_voice(monkeypatch, FakeSTT(heard("What's my current balance?")), FakeTTS())

    assert await voice_turn.main([str(question), "--out", str(answer)]) == 0

    out = capsys.readouterr().out
    assert 'heard:  "What\'s my current balance?" (en, confidence 0.80)' in out
    assert "answer: Here are your current balances:" in out
    assert answer.read_bytes()[:4] == b"RIFF"


@pytest.mark.usefixtures("seeded_db")
async def test_voice_turn_reports_problems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import voice_turn
    from tests.fakes import wav_bytes

    question = tmp_path / "q.wav"
    question.write_bytes(wav_bytes())

    use_voice(monkeypatch, None, None)
    assert await voice_turn.main([str(question)]) == 1
    assert "Voice input is disabled" in capsys.readouterr().err

    use_voice(monkeypatch, FakeSTT(heard("hi")), None)
    assert await voice_turn.main([str(question), "--customer", "CUST-9999"]) == 1
    assert "No mock customer CUST-9999" in capsys.readouterr().err

    not_audio = tmp_path / "notes.txt"
    not_audio.write_text("this is not audio" * 10)
    use_voice(monkeypatch, FakeSTT(), None)
    assert await voice_turn.main([str(not_audio)]) == 1
    assert "unsupported_audio" in capsys.readouterr().err

    use_voice(monkeypatch, FakeSTT(heard("hello")), FakeTTS(error=SpeechUnavailableError("down")))
    assert await voice_turn.main([str(question)]) == 0
    assert "speech: not available (voice_unavailable)" in capsys.readouterr().out
