"""Ask the assistant with a recorded question; get the answer as text and as speech.

    python -m scripts.voice_turn question.wav                      # writes answer.wav
    python -m scripts.voice_turn question.webm --out reply.wav --customer CUST-0003 --language en

Needs STT_PROVIDER / TTS_PROVIDER configured (e.g. faster_whisper + piper, see .env.example).
The bank is the configured database; signing in by customer number is a MOCK login.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.factory import open_assistant
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine
from app.models import User
from app.voice.errors import VoiceError
from app.voice.factory import create_stt, create_tts
from app.voice.pipeline import VoicePipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One voice turn with BankAI (MOCK data).")
    parser.add_argument("audio", type=Path, help="recorded question (WAV, MP3, OGG, WebM, M4A, FLAC)")
    parser.add_argument(
        "--out", type=Path, default=Path("answer.wav"), help="where to save the spoken answer"
    )
    parser.add_argument("--customer", default="CUST-0001", help="mock customer number to act as")
    parser.add_argument("--language", choices=["en", "ka"], default=None, help="skip language detection")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING", "console")
    stt, tts = create_stt(settings), create_tts(settings)
    if stt is None:
        print("Voice input is disabled: set STT_PROVIDER (see .env.example).", file=sys.stderr)
        return 1

    engine = create_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            user = await session.scalar(select(User).where(User.customer_number == args.customer.upper()))
        if user is None:
            print(f"No mock customer {args.customer}. Run: python -m scripts.seed_db", file=sys.stderr)
            return 1
        async with open_assistant(settings, sessions) as assistant:
            pipeline = VoicePipeline(
                assistant,
                stt,
                tts,
                min_confidence=settings.stt_min_confidence,
                max_audio_bytes=settings.voice_max_audio_bytes,
                max_audio_seconds=settings.voice_max_audio_seconds,
                tts_max_chars=settings.tts_max_chars,
            )
            turn = await pipeline.converse(args.audio.read_bytes(), user_id=user.id, language=args.language)
    except VoiceError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await stt.aclose()
        if tts is not None:
            await tts.aclose()
        await engine.dispose()

    heard = turn.transcription
    print(f"heard:  {heard.text!r} ({heard.language}, confidence {heard.confidence:.2f})")
    print(f"answer: {turn.response_text}")
    if turn.speech is not None:
        args.out.write_bytes(turn.speech.audio)
        print(f"speech: {args.out} ({turn.speech.duration_s} s, voice {turn.speech.voice})")
    elif turn.errors:
        print(f"speech: not available ({', '.join(turn.errors)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
