"""Test doubles."""

from collections.abc import Sequence
from typing import Any

from app.llm import ChatMessage, LLMCompletion, LLMError


class ScriptedLLM:
    """An LLMClient that replays scripted replies (or raises scripted errors) in order.

    Every call is recorded, so tests can assert what the agent asked the model.
    """

    def __init__(self, *replies: str | LLMError) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "scripted"

    async def aclose(self) -> None:
        self.closed = True

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        self.calls.append({"messages": list(messages), "json_schema": json_schema, "max_tokens": max_tokens})
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self._replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        return LLMCompletion(text=reply, model="scripted", latency_ms=1.0)


# --- voice ----------------------------------------------------------------------------------------


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A valid mono 16-bit WAV of silence."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


class FakeSTT:
    """A SpeechToText that "hears" scripted transcriptions (or raises scripted errors)."""

    def __init__(self, *results: "Any") -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "fake-stt"

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> Any:
        self.calls.append({"bytes": len(audio), "language": language})
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        self.closed = True


class FakeTTS:
    """A TextToSpeech that returns a short WAV (or raises a scripted error)."""

    def __init__(
        self, error: Exception | None = None, languages: frozenset[str] = frozenset({"en", "ka"})
    ) -> None:
        self._error = error
        self._languages = languages
        self.spoken: list[tuple[str, str]] = []

    @property
    def languages(self) -> frozenset[str]:
        return self._languages

    async def synthesize(self, text: str, *, language: str) -> Any:
        from app.voice.tts import SpeechAudio

        self.spoken.append((text, language))
        if self._error is not None:
            raise self._error
        return SpeechAudio(wav_bytes(0.5), "audio/wav", language, f"fake-{language}", 0.5, 1.0)

    async def aclose(self) -> None:
        return None
