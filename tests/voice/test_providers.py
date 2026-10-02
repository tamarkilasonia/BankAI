"""STT/TTS providers: OpenAI-compatible ones against a mocked HTTP API, local ones' failure paths.

The local models themselves are exercised in tests/voice/test_local_models.py
(skipped when the optional packages or models are missing).
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

from app.core.config import Settings
from app.voice.errors import (
    InvalidAudioError,
    SpeechUnavailableError,
    SynthesisError,
    TranscriptionError,
    UnsupportedLanguageError,
)
from app.voice.factory import create_stt, create_tts
from app.voice.stt import FasterWhisperSTT, OpenAICompatibleSTT, confidence_from, echoes_prompt
from app.voice.tts import OpenAICompatibleTTS, PiperTTS
from tests.fakes import wav_bytes

Handler = Callable[[httpx2.Request], httpx2.Response]


def mock(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def verbose_json(text: str, language: str, logprobs: list[float]) -> dict[str, Any]:
    return {
        "text": text,
        "language": language,
        "duration": 2.5,
        "segments": [
            {
                "id": i, "seek": 0, "start": float(i), "end": float(i + 1), "text": text, "tokens": [1],
                "temperature": 0.0, "avg_logprob": lp, "compression_ratio": 1.0, "no_speech_prob": 0.01,
            }
            for i, lp in enumerate(logprobs)
        ],
    }  # fmt: skip


# --- confidence ---------------------------------------------------------------------------------


def test_confidence_is_a_duration_weighted_probability() -> None:
    assert confidence_from([]) == 0.0
    assert confidence_from([(0.0, 1.0)]) == 1.0
    assert confidence_from([(-0.3, 1.0)]) == 0.741  # exp(-0.3)
    assert confidence_from([(-0.1, 3.0), (-2.0, 1.0)]) == 0.563  # exp((-0.3 - 2.0) / 4)
    assert confidence_from([(-0.5, 0.0)]) == 0.0  # no duration: nothing was heard


# --- OpenAI-compatible speech-to-text ---------------------------------------------------------------


async def test_openai_stt_request_and_mapping() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx2.Response(200, json=verbose_json("What's my balance?", "english", [-0.2, -0.4]))

    stt = OpenAICompatibleSTT(base_url="http://stt.test/v1", prompt="lari (GEL)", http_client=mock(handler))
    result = await stt.transcribe(wav_bytes(), language=None)

    assert seen["url"] == "http://stt.test/v1/audio/transcriptions"
    assert b'name="response_format"' in seen["body"] and b"verbose_json" in seen["body"]
    assert b'name="language"' not in seen["body"]  # not sent: the service detects it
    assert b'name="prompt"' in seen["body"] and b"lari (GEL)" in seen["body"]
    assert (result.text, result.language, result.confidence, result.duration_s) == (
        "What's my balance?",
        "en",
        0.741,
        2.5,
    )
    assert result.model == "openai-compatible/whisper-1"
    await stt.aclose()


async def test_openai_stt_sends_the_language_hint() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert b'name="language"' in request.content and b"\r\n\r\nka\r\n" in request.content
        return httpx2.Response(200, json=verbose_json("...", "georgian", []))

    result = await OpenAICompatibleSTT(base_url="http://stt.test/v1", http_client=mock(handler)).transcribe(
        wav_bytes(), language="ka"
    )
    assert (result.language, result.confidence) == ("ka", 0.0)


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (
            lambda r: httpx2.Response(400, json={"error": {"message": "Invalid file format."}}),
            InvalidAudioError,
        ),
        (lambda r: httpx2.Response(503, json={"error": {"message": "overloaded"}}), SpeechUnavailableError),
        (lambda r: (_ for _ in ()).throw(httpx2.ConnectError("refused")), SpeechUnavailableError),
        (lambda r: (_ for _ in ()).throw(httpx2.ReadTimeout("slow")), SpeechUnavailableError),
    ],
)
async def test_openai_stt_errors(handler: Handler, error: type[Exception]) -> None:
    stt = OpenAICompatibleSTT(base_url="http://stt.test/v1", http_client=mock(handler))
    with pytest.raises(error):
        await stt.transcribe(wav_bytes())


def test_openai_needs_a_key_for_api_openai_com() -> None:
    with pytest.raises(SpeechUnavailableError, match="STT_API_KEY"):
        OpenAICompatibleSTT()
    with pytest.raises(SpeechUnavailableError, match="TTS_API_KEY"):
        OpenAICompatibleTTS()


# --- OpenAI-compatible text-to-speech ----------------------------------------------------------------


async def test_openai_tts_request_and_audio() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, content=wav_bytes(1.5), headers={"content-type": "audio/wav"})

    tts = OpenAICompatibleTTS(base_url="http://tts.test/v1", http_client=mock(handler))
    speech = await tts.synthesize("Hello", language="en")

    assert seen["url"] == "http://tts.test/v1/audio/speech"
    assert seen["body"] == {
        "input": "Hello",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "response_format": "wav",
    }
    assert (speech.content_type, speech.duration_s, speech.voice, speech.language) == (
        "audio/wav",
        1.5,
        "alloy",
        "en",
    )
    await tts.aclose()


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda r: httpx2.Response(500, json={"error": {"message": "boom"}}), SynthesisError),
        (lambda r: (_ for _ in ()).throw(httpx2.ConnectError("refused")), SpeechUnavailableError),
    ],
)
async def test_openai_tts_errors(handler: Handler, error: type[Exception]) -> None:
    with pytest.raises(error):
        await OpenAICompatibleTTS(base_url="http://tts.test/v1", http_client=mock(handler)).synthesize(
            "Hello", language="en"
        )


async def test_tts_rejects_languages_without_a_voice() -> None:
    tts = OpenAICompatibleTTS(base_url="http://tts.test/v1", languages=frozenset({"en"}))
    with pytest.raises(UnsupportedLanguageError):
        await tts.synthesize("Bonjour", language="fr")
    with pytest.raises(UnsupportedLanguageError):
        await PiperTTS({"en": "en_US-lessac-medium"}).synthesize("Bonjour", language="fr")


# --- local providers: what happens when they cannot run ------------------------------------------------


async def test_missing_whisper_model_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper")
    stt = FasterWhisperSTT("small", model_dir=tmp_path)  # nothing downloaded here
    with pytest.raises(SpeechUnavailableError, match="not downloaded"):
        await stt.transcribe(wav_bytes())


async def test_missing_piper_voice_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("piper")
    with pytest.raises(SpeechUnavailableError, match="not downloaded"):
        await PiperTTS({"en": "en_US-lessac-medium"}, voices_dir=tmp_path).synthesize("Hi", language="en")


async def test_missing_packages_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes the import fail
    monkeypatch.setitem(sys.modules, "piper", None)
    with pytest.raises(SpeechUnavailableError, match=r"requirements-voice\.txt"):
        await FasterWhisperSTT().transcribe(wav_bytes())
    with pytest.raises(SpeechUnavailableError, match=r"requirements-voice\.txt"):
        await PiperTTS({"en": "x"}).synthesize("Hi", language="en")


class BrokenVoice:
    def synthesize_wav(self, text: str, wav: object) -> None:
        raise RuntimeError("onnx exploded")


async def test_piper_failure_becomes_a_voice_error() -> None:
    tts = PiperTTS({"en": "en_US-lessac-medium"})
    tts._loaded["en"] = BrokenVoice()
    with pytest.raises(SynthesisError):
        await tts.synthesize("Hi", language="en")
    await tts.aclose()


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: records how it was called."""

    def __init__(
        self, detected: str = "en", text: str = "What's my balance?", error: Exception | None = None
    ) -> None:
        self.detected, self.text, self.error = detected, text, error
        self.calls: list[dict[str, Any]] = []

    def detect_language(self, samples: object, **kwargs: object) -> tuple[str, float, list[object]]:
        self.calls.append({"detect": True})
        return self.detected, 0.93, []

    def transcribe(self, samples: object, **kwargs: Any) -> tuple[list[Any], Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        segment = SimpleNamespace(text=f" {self.text}", avg_logprob=-0.2, start=0.0, end=1.0)
        return [segment], SimpleNamespace(language=kwargs["language"], duration=1.0)


def whisper(model: FakeWhisperModel, prompt: str = "lari (GEL) banking") -> FasterWhisperSTT:
    pytest.importorskip("faster_whisper")  # audio decoding comes from the package
    stt = FasterWhisperSTT(prompt=prompt)
    stt._model = model
    return stt


async def test_whisper_uses_the_vocabulary_hint_only_for_english() -> None:
    english, georgian = FakeWhisperModel("en"), FakeWhisperModel("ka", text="რამდენი")

    heard_en = await whisper(english).transcribe(wav_bytes())
    heard_ka = await whisper(georgian).transcribe(wav_bytes())

    assert english.calls[1]["initial_prompt"] == "lari (GEL) banking" and english.calls[1]["language"] == "en"
    assert georgian.calls[1]["initial_prompt"] is None
    assert (heard_en.text, heard_en.language, heard_en.language_probability) == (
        "What's my balance?",
        "en",
        0.93,
    )
    assert heard_en.confidence == 0.819  # exp(-0.2)
    assert heard_ka.language == "ka"


async def test_a_given_language_skips_detection() -> None:
    model = FakeWhisperModel()
    stt = whisper(model)
    heard = await stt.transcribe(wav_bytes(), language="en")
    assert "detect" not in model.calls[0] and heard.language_probability is None
    await stt.aclose()
    assert stt._model is None  # the model is released


async def test_an_echo_of_the_prompt_counts_as_nothing_heard() -> None:
    """Measured failure: unintelligible speech came back as the prompt itself, with confidence ~0.7."""
    model = FakeWhisperModel("en", text="A customer asks a banking assistant about lari (GEL) banking stuff.")
    heard = await whisper(model, prompt="A customer asks a banking assistant about lari (GEL)").transcribe(
        wav_bytes()
    )
    assert (heard.text, heard.confidence) == ("", 0.0)
    assert not echoes_prompt("What's my balance in lari?", "A customer asks a banking assistant about lari")
    assert not echoes_prompt("anything", None)


async def test_whisper_failures_become_voice_errors() -> None:
    decode_error = type("InvalidDataError", (Exception,), {"__module__": "av.error"})("bad container")
    with pytest.raises(InvalidAudioError):
        await whisper(FakeWhisperModel(error=decode_error)).transcribe(wav_bytes())
    with pytest.raises(TranscriptionError):
        await whisper(FakeWhisperModel(error=ValueError("model"))).transcribe(wav_bytes())
    with pytest.raises(InvalidAudioError):  # real decoding of something that is not audio
        await whisper(FakeWhisperModel()).transcribe(b"OggS" + b"\x00not really ogg" * 20)


# --- configuration ----------------------------------------------------------------------------------------


def test_factory_follows_the_settings() -> None:
    off = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (create_stt(off), create_tts(off)) == (None, None)

    local = Settings(_env_file=None, stt_provider="faster_whisper", tts_provider="piper")  # type: ignore[call-arg]
    assert isinstance(create_stt(local), FasterWhisperSTT)
    tts = create_tts(local)
    assert isinstance(tts, PiperTTS) and tts.languages == {"en", "ka"}

    remote = Settings(
        _env_file=None,  # type: ignore[call-arg]
        stt_provider="openai_compatible",
        stt_model="whisper-1",
        stt_api_key="k" * 10,
        tts_provider="openai_compatible",
        tts_base_url="http://tts.test/v1",
    )
    assert isinstance(create_stt(remote), OpenAICompatibleSTT)
    assert isinstance(create_tts(remote), OpenAICompatibleTTS)


def test_empty_voice_keys_mean_not_configured() -> None:
    settings = Settings(_env_file=None, stt_api_key="", tts_api_key="")  # type: ignore[call-arg]
    assert (settings.stt_api_key, settings.tts_api_key) == (None, None)


async def test_openai_edge_cases_without_wav_or_language() -> None:
    tts = OpenAICompatibleTTS(
        base_url="http://tts.test/v1",
        http_client=mock(lambda r: httpx2.Response(200, content=b"ID3 not a wav")),
    )
    speech = await tts.synthesize("Hello", language="en")
    assert speech.duration_s is None  # length unknown, audio still returned
    assert tts.languages == {"en", "ka"}

    body = verbose_json("hello", "english", [-0.1]) | {"language": ""}
    stt = OpenAICompatibleSTT(
        base_url="http://stt.test/v1", http_client=mock(lambda r: httpx2.Response(200, json=body))
    )
    assert (await stt.transcribe(wav_bytes())).language is None
