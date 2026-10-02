"""Voice errors. Each has a stable ``code`` (the API maps codes to HTTP statuses) and a
message written for the customer - never a stack trace or a library's internals."""

from typing import ClassVar


class VoiceError(Exception):
    code: ClassVar[str] = "voice_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidAudioError(VoiceError):
    """Empty, truncated or corrupted audio."""

    code = "invalid_audio"


class UnsupportedAudioError(VoiceError):
    """A file that is not audio, or an audio format we do not accept."""

    code = "unsupported_audio"


class AudioTooLargeError(VoiceError):
    code = "audio_too_large"


class AudioTooLongError(VoiceError):
    code = "audio_too_long"


class UnsupportedLanguageError(VoiceError):
    code = "unsupported_language"


class TextTooLongError(VoiceError):
    code = "text_too_long"


class SpeechUnavailableError(VoiceError):
    """The speech service or model is not available (not installed, not downloaded, unreachable)."""

    code = "voice_unavailable"


class TranscriptionError(VoiceError):
    code = "stt_failed"


class SynthesisError(VoiceError):
    code = "tts_failed"
