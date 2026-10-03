"""The real local models: Whisper (faster-whisper) and Piper.

Marked ``voice`` and skipped when the optional packages or the downloaded models are
missing (CI does not install them). Run with: pytest -m voice
"""

import io
import os
import re
from pathlib import Path

import pytest

from app.core.config import Settings
from app.voice.errors import InvalidAudioError
from app.voice.pipeline import VoicePipeline
from app.voice.stt import FasterWhisperSTT
from app.voice.tts import PiperTTS

pytestmark = pytest.mark.voice

settings = Settings(_env_file=None)  # type: ignore[call-arg]
VOICES = {"en": settings.tts_voice_en, "ka": settings.tts_voice_ka}
_GEORGIAN = re.compile(f"[{chr(0x10A0)}-{chr(0x10FF)}]")


def _models_available() -> bool:
    try:
        import av  # noqa: F401
        import faster_whisper  # noqa: F401
        import piper  # noqa: F401
    except ImportError:
        return False
    voices = all((settings.tts_voices_dir / f"{v}.onnx").exists() for v in VOICES.values())
    return voices and any(Path(settings.stt_model_dir).glob(f"models--*faster-whisper-{settings.stt_model}"))


if not _models_available():
    if os.getenv("REQUIRE_VOICE") == "1":  # the live-models workflow: missing models are a failure
        pytest.fail("local voice models not installed (REQUIRE_VOICE=1)", pytrace=False)
    pytest.skip(
        "local voice models not installed (requirements-voice.txt + download)", allow_module_level=True
    )

import av  # noqa: E402 - only after the skip

STT = FasterWhisperSTT(settings.stt_model, model_dir=settings.stt_model_dir)
TTS = PiperTTS(VOICES, voices_dir=settings.tts_voices_dir)


def transcode(wav: bytes, container: str, codec: str, rate: int) -> bytes:
    """Re-encode WAV the way browsers record (WebM/Opus, Ogg/Opus, MP4/AAC)."""
    output = io.BytesIO()
    with av.open(io.BytesIO(wav)) as source, av.open(output, "w", format=container) as target:
        stream = target.add_stream(codec, rate=rate)
        resampler = av.AudioResampler(format=stream.format, layout="mono", rate=rate)
        for frame in source.decode(audio=0):
            for resampled in resampler.resample(frame):
                target.mux(stream.encode(resampled))
        for resampled in resampler.resample(None):
            target.mux(stream.encode(resampled))
        target.mux(stream.encode(None))
    return output.getvalue()


async def test_english_round_trip() -> None:
    speech = await TTS.synthesize("What is my current balance?", language="en")
    heard = await STT.transcribe(speech.audio)

    assert speech.duration_s and speech.duration_s > 0.5
    assert heard.text == "What is my current balance?"
    assert heard.language == "en" and heard.confidence >= 0.6


@pytest.mark.parametrize(
    ("container", "codec", "rate"),
    [("webm", "libopus", 48000), ("ogg", "libopus", 48000), ("mp4", "aac", 44100)],
)
async def test_browser_recording_formats_are_decoded(container: str, codec: str, rate: int) -> None:
    speech = await TTS.synthesize("Show my recent transactions.", language="en")
    recording = transcode(speech.audio, container, codec, rate)

    heard = await STT.transcribe(recording)

    assert "recent transactions" in heard.text.lower()


async def test_a_damaged_recording_is_invalid_audio() -> None:
    with pytest.raises(InvalidAudioError):
        await STT.transcribe(b"\x1a\x45\xdf\xa3" + b"\x00garbage" * 200)


async def test_silence_is_heard_as_nothing() -> None:
    from tests.fakes import wav_bytes

    heard = await STT.transcribe(wav_bytes(2.0))
    assert heard.text == "" and heard.confidence == 0.0


async def test_georgian_voice_speaks_georgian() -> None:
    speech = await TTS.synthesize("რამდენი თანხა მაქვს ანგარიშზე?", language="ka")
    assert speech.voice == settings.tts_voice_ka and speech.duration_s and speech.duration_s > 1.0


async def test_garbled_georgian_is_never_acted_on() -> None:
    """Whisper often writes Georgian in Latin letters with low confidence (a documented limitation).

    The safety property: either the transcript is real Georgian, or the pipeline does not act on it.
    """
    speech = await TTS.synthesize("რამდენი თანხა მაქვს ანგარიშზე?", language="ka")
    heard = await STT.transcribe(speech.audio)
    pipeline = VoicePipeline(None, STT, None)  # type: ignore[arg-type] - the agent is not reached here

    understood = pipeline.why_not_understood(heard, None) is None
    assert not understood or _GEORGIAN.search(heard.text)
