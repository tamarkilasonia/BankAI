"""Text-to-speech behind one interface.

* ``PiperTTS`` - local neural voices (ONNX), one voice per language. This project uses
  ``en_US-lessac-medium`` and ``ka_GE-natia-medium`` (Piper has a Georgian voice).
* ``OpenAICompatibleTTS`` - any OpenAI-style ``/audio/speech`` endpoint.

Both return WAV bytes, which every browser can play.
"""

import asyncio
import io
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.voice.errors import SpeechUnavailableError, SynthesisError, UnsupportedLanguageError

_PLACEHOLDER_KEY = "not-needed-for-local-servers"


@dataclass(frozen=True)
class SpeechAudio:
    audio: bytes
    content_type: str
    language: str
    voice: str
    duration_s: float | None
    latency_ms: float


class TextToSpeech(Protocol):
    @property
    def languages(self) -> frozenset[str]: ...

    async def synthesize(self, text: str, *, language: str) -> SpeechAudio: ...

    async def aclose(self) -> None: ...


def _wav_seconds(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as wav:
            seconds: float = wav.getnframes() / wav.getframerate()
        return round(seconds, 2)
    except (wave.Error, EOFError, RuntimeError, ValueError, ZeroDivisionError):
        return None


class PiperTTS:
    def __init__(self, voices: dict[str, str], *, voices_dir: Path = Path("models/piper")) -> None:
        """``voices``: language -> Piper voice name (e.g. ``{"en": "en_US-lessac-medium"}``)."""
        self._voices = voices
        self._dir = voices_dir
        self._loaded: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(self._voices)

    def _voice(self, language: str) -> Any:
        if language not in self._loaded:
            try:
                from piper import PiperVoice  # optional dependency, imported on first use
            except ImportError as exc:
                raise SpeechUnavailableError(
                    "Speech synthesis is not installed (pip install -r requirements-voice.txt)."
                ) from exc
            path = self._dir / f"{self._voices[language]}.onnx"
            if not path.exists():
                raise SpeechUnavailableError(
                    f"Voice {self._voices[language]} is not downloaded "
                    "(python -m scripts.download_voice_models)."
                )
            self._loaded[language] = PiperVoice.load(path)
        return self._loaded[language]

    def _run(self, text: str, language: str) -> SpeechAudio:
        started = time.perf_counter()
        voice = self._voice(language)
        buffer = io.BytesIO()
        try:
            with wave.open(buffer, "wb") as wav:
                voice.synthesize_wav(text, wav)
        except Exception as exc:
            raise SynthesisError("Speech synthesis failed.") from exc
        audio = buffer.getvalue()
        return SpeechAudio(
            audio=audio,
            content_type="audio/wav",
            language=language,
            voice=self._voices[language],
            duration_s=_wav_seconds(audio),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def synthesize(self, text: str, *, language: str) -> SpeechAudio:
        if language not in self._voices:
            raise UnsupportedLanguageError(f"No voice for language '{language}'.")
        async with self._lock:
            return await asyncio.to_thread(self._run, text, language)

    async def aclose(self) -> None:
        self._loaded.clear()


class OpenAICompatibleTTS:
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        languages: frozenset[str] = frozenset({"en", "ka"}),
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if base_url is None and not api_key:
            raise SpeechUnavailableError("TTS_API_KEY is required when calling api.openai.com.")
        self._model = model
        self._voice_name = voice
        self._languages = languages
        self._client = AsyncOpenAI(
            api_key=api_key or _PLACEHOLDER_KEY,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=1,
            http_client=http_client,
        )

    @property
    def languages(self) -> frozenset[str]:
        return self._languages

    async def synthesize(self, text: str, *, language: str) -> SpeechAudio:
        if language not in self._languages:
            raise UnsupportedLanguageError(f"No voice for language '{language}'.")
        started = time.perf_counter()
        try:
            response = await self._client.audio.speech.create(
                model=self._model, voice=self._voice_name, input=text, response_format="wav"
            )
        except (APITimeoutError, APIConnectionError) as exc:
            raise SpeechUnavailableError("Speech synthesis is unreachable.") from exc
        except APIStatusError as exc:
            raise SynthesisError(f"Speech synthesis returned HTTP {exc.status_code}.") from exc
        audio = response.content
        return SpeechAudio(
            audio=audio,
            content_type="audio/wav",
            language=language,
            voice=self._voice_name,
            duration_s=_wav_seconds(audio),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def aclose(self) -> None:
        await self._client.close()
