"""Measure the voice round trip (text -> Piper TTS -> Whisper STT -> text) in English and Georgian.

    python -m scripts.evaluate_voice                       # writes docs/evaluation/voice.md
    python -m scripts.evaluate_voice --output report.md

Uses the configured local models regardless of STT_PROVIDER/TTS_PROVIDER (needs
requirements-voice.txt and python -m scripts.download_voice_models).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evaluation.voice import run_round_trips, to_markdown
from app.voice.errors import VoiceError
from app.voice.stt import FasterWhisperSTT
from app.voice.tts import PiperTTS

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "evaluation" / "voice.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice round-trip evaluation (MOCK banking questions).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("ERROR", "console")
    voices = {"en": settings.tts_voice_en, "ka": settings.tts_voice_ka}
    tts = PiperTTS(voices, voices_dir=settings.tts_voices_dir)
    stt = FasterWhisperSTT(settings.stt_model, model_dir=settings.stt_model_dir, prompt=settings.stt_prompt)
    try:
        report = await run_round_trips(tts, stt, min_confidence=settings.stt_min_confidence, voices=voices)
    except VoiceError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await tts.aclose()
        await stt.aclose()

    args.output.write_text(to_markdown(report), encoding="utf-8")
    for language, stats in report.languages.items():
        print(
            f"{language}: WER {stats.mean_wer:.1%}, exact {stats.exact}/{stats.samples}, "
            f"acted on {stats.acted_on}/{stats.samples}, false accepts {stats.false_accepts}, "
            f"false rejects {stats.false_rejects}, mean confidence {stats.mean_confidence}"
        )
    if report.threshold_window:
        low, high = report.threshold_window
        print(f"bad transcripts score up to {low}, good ones from {high}")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
