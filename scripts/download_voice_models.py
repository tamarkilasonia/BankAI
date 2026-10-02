"""Download the local speech models (once; afterwards everything runs offline).

    python -m scripts.download_voice_models            # models named in the settings
    python -m scripts.download_voice_models --whisper base

* Whisper (speech-to-text, faster-whisper / CTranslate2): STT_MODEL into STT_MODEL_DIR
  (``small`` is ~460 MB).
* Piper voices (text-to-speech): TTS_VOICE_EN and TTS_VOICE_KA into TTS_VOICES_DIR (~60 MB each).

Requires the optional voice packages: pip install -r requirements-voice.txt
"""

import argparse
import sys

from app.core.config import get_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download Whisper and Piper models.")
    parser.add_argument("--whisper", default=settings.stt_model, help="Whisper model size, e.g. small")
    parser.add_argument(
        "--voice",
        action="append",
        default=None,
        help="Piper voice (repeatable); default: the configured voices",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    try:
        from faster_whisper import download_model  # optional dependencies
        from piper.download_voices import download_voice
    except ImportError:
        print("Voice packages missing: pip install -r requirements-voice.txt", file=sys.stderr)
        return 1

    settings.stt_model_dir.mkdir(parents=True, exist_ok=True)
    path = download_model(args.whisper, cache_dir=str(settings.stt_model_dir))
    print(f"Whisper {args.whisper}: {path}")

    settings.tts_voices_dir.mkdir(parents=True, exist_ok=True)
    for voice in args.voice or [settings.tts_voice_en, settings.tts_voice_ka]:
        download_voice(voice, settings.tts_voices_dir)
        print(f"Piper voice {voice}: {settings.tts_voices_dir / voice}.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
