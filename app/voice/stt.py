"""Speech-to-text behind one interface.

* ``FasterWhisperSTT`` - OpenAI's Whisper model running locally (CTranslate2, int8 on CPU).
  Free and offline once the model is downloaded; ~2 s for a short question with ``small``.
* ``OpenAICompatibleSTT`` - any OpenAI-style ``/audio/transcriptions`` endpoint.

Every transcription carries a **confidence** (from the model's token probabilities)
and the detected language, because a banking assistant must not act on a guess:
the pipeline asks the customer to repeat when the confidence is low.

Georgian is a known weak spot of Whisper: measured in this project, ``small`` writes
Georgian speech in Latin letters with low confidence, and ``large-v3-turbo`` mistakes it
for Persian. See docs/ARCHITECTURE.md (voice) for the numbers.
"""

import asyncio
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, omit

from app.voice.errors import InvalidAudioError, SpeechUnavailableError, TranscriptionError

# Language names returned by the OpenAI API -> ISO 639-1 codes.
_LANGUAGE_CODES = {"english": "en", "georgian": "ka"}
_PLACEHOLDER_KEY = "not-needed-for-local-servers"


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None  # ISO 639-1 ("en", "ka"), detected or as requested
    language_probability: float | None  # None when the language was given, not detected
    confidence: float  # 0..1: geometric mean of token probabilities (exp of the mean log-prob)
    duration_s: float
    model: str
    latency_ms: float


class SpeechToText(Protocol):
    @property
    def name(self) -> str: ...

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> Transcription: ...

    async def aclose(self) -> None: ...


def _ngrams(text: str, size: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def echoes_prompt(text: str, prompt: str | None) -> bool:
    """Whisper's known failure: on speech it cannot understand it repeats its prompt, confidently."""
    return bool(prompt) and bool(_ngrams(text) & _ngrams(prompt or ""))


def confidence_from(segments: list[tuple[float, float]]) -> float:
    """``(avg_logprob, duration)`` per segment -> one duration-weighted probability in 0..1."""
    total = sum(duration for _, duration in segments)
    if not segments or total <= 0:
        return 0.0
    mean_logprob = sum(logprob * duration for logprob, duration in segments) / total
    return round(min(1.0, math.exp(mean_logprob)), 3)


class FasterWhisperSTT:
    def __init__(
        self,
        model: str = "small",
        *,
        model_dir: Path = Path("models/whisper"),
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        prompt: str = "",
    ) -> None:
        self._model_name = model
        self._prompt = prompt or None
        self._model_dir = model_dir
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model: Any = None
        # One transcription at a time: the model is CPU-bound and loaded once.
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return f"faster-whisper/{self._model_name}"

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # optional dependency, imported on first use
            except ImportError as exc:
                raise SpeechUnavailableError(
                    "Speech recognition is not installed (pip install -r requirements-voice.txt)."
                ) from exc
            try:
                # Local files only: audio and model never touch the network at runtime.
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=str(self._model_dir),
                    local_files_only=True,
                )
            except Exception as exc:
                raise SpeechUnavailableError(
                    f"Whisper model '{self._model_name}' is not downloaded "
                    "(python -m scripts.download_voice_models)."
                ) from exc
        return self._model

    def _run(self, audio: bytes, language: str | None) -> Transcription:
        started = time.perf_counter()
        model = self._load()
        try:
            from faster_whisper import decode_audio  # optional dependency, imported on first use

            samples = decode_audio(io.BytesIO(audio))
            probability = None
            if language is None:
                language, probability, _ = model.detect_language(samples, vad_filter=True)
            # The vocabulary hint is English: given to Georgian speech, Whisper "hears" the hint itself
            # (measured: 5 of 12 Georgian questions came back as the English prompt, confidently).
            prompt = self._prompt if language == "en" else None
            # VAD skips silence, so a silent recording gives an empty transcript, not hallucinated words.
            segments, info = model.transcribe(
                samples, language=language, beam_size=self._beam_size, vad_filter=True, initial_prompt=prompt
            )
            segments = list(segments)  # the generator does the actual decoding
        except Exception as exc:
            if type(exc).__module__.startswith("av"):  # PyAV could not decode the container
                raise InvalidAudioError("The recording could not be decoded.") from exc
            raise TranscriptionError("Speech recognition failed.") from exc
        text = " ".join(s.text.strip() for s in segments).strip()
        if echoes_prompt(text, prompt):
            text, segments = "", []  # treat as "nothing understood" - never act on an echo
        return Transcription(
            text=text,
            language=info.language,
            language_probability=None if probability is None else round(probability, 3),
            confidence=confidence_from([(s.avg_logprob, s.end - s.start) for s in segments]),
            duration_s=round(info.duration, 2),
            model=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> Transcription:
        async with self._lock:
            return await asyncio.to_thread(self._run, audio, language)

    async def aclose(self) -> None:
        self._model = None


class OpenAICompatibleSTT:
    def __init__(
        self,
        *,
        model: str = "whisper-1",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        prompt: str = "",
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if base_url is None and not api_key:
            raise SpeechUnavailableError("STT_API_KEY is required when calling api.openai.com.")
        self._model = model
        self._prompt = prompt
        self._client = AsyncOpenAI(
            api_key=api_key or _PLACEHOLDER_KEY,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=1,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return f"openai-compatible/{self._model}"

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> Transcription:
        started = time.perf_counter()
        try:
            result = await self._client.audio.transcriptions.create(
                file=("audio", audio),
                model=self._model,
                response_format="verbose_json",
                language=language or omit,
                prompt=self._prompt or omit,
            )
        except APITimeoutError as exc:
            raise SpeechUnavailableError("Speech recognition did not answer in time.") from exc
        except APIConnectionError as exc:
            raise SpeechUnavailableError("Speech recognition is unreachable.") from exc
        except APIStatusError as exc:
            if exc.status_code == 400:
                raise InvalidAudioError("The speech service could not read the recording.") from exc
            raise SpeechUnavailableError(f"Speech recognition returned HTTP {exc.status_code}.") from exc
        segments = [(s.avg_logprob, s.end - s.start) for s in (result.segments or [])]
        detected = _LANGUAGE_CODES.get(result.language.lower(), result.language) if result.language else None
        return Transcription(
            text=result.text.strip(),
            language=language or detected,
            language_probability=None,  # not reported by the API
            confidence=confidence_from(segments) if segments else 0.0,
            duration_s=round(float(result.duration), 2),
            model=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def aclose(self) -> None:
        await self._client.close()
