"""Checking uploaded audio before anything expensive happens.

The format is detected from the first bytes of the file ("magic numbers"), not
from its name or the Content-Type header - both are chosen by the client and
cannot be trusted. Browsers record WebM/Opus (Chrome, Firefox) or MP4/AAC
(Safari); WAV, MP3, OGG and FLAC are accepted too.
"""

import io
import wave
from dataclasses import dataclass

from app.voice.errors import AudioTooLargeError, AudioTooLongError, InvalidAudioError, UnsupportedAudioError

MIN_AUDIO_BYTES = 44  # a WAV header alone is 44 bytes


@dataclass(frozen=True)
class AudioFormat:
    name: str  # wav | mp3 | ogg | webm | mp4 | flac
    content_type: str


WAV = AudioFormat("wav", "audio/wav")
FORMATS = {
    "wav": WAV,
    "mp3": AudioFormat("mp3", "audio/mpeg"),
    "ogg": AudioFormat("ogg", "audio/ogg"),
    "webm": AudioFormat("webm", "audio/webm"),
    "mp4": AudioFormat("mp4", "audio/mp4"),
    "flac": AudioFormat("flac", "audio/flac"),
}
SUPPORTED = "WAV, MP3, OGG, WebM, MP4/M4A or FLAC"


def detect_format(data: bytes) -> AudioFormat | None:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return FORMATS["wav"]
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return FORMATS["mp3"]
    if data[:4] == b"OggS":
        return FORMATS["ogg"]
    if data[:4] == b"\x1a\x45\xdf\xa3":  # EBML header: WebM / Matroska
        return FORMATS["webm"]
    if data[4:8] == b"ftyp":
        return FORMATS["mp4"]
    if data[:4] == b"fLaC":
        return FORMATS["flac"]
    return None


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data)) as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
    except (wave.Error, EOFError, RuntimeError, ValueError) as exc:  # the wave module raises all of these
        raise InvalidAudioError("The WAV file is damaged or incomplete.") from exc
    if not frames or not rate:
        raise InvalidAudioError("The recording is empty.")
    seconds: float = frames / rate
    return seconds


def validate_audio(data: bytes, *, max_bytes: int, max_seconds: float) -> AudioFormat:
    """Raise a ``VoiceError`` unless ``data`` is acceptable audio. Returns its format.

    WAV length is checked here; for compressed formats it is checked after decoding.
    """
    if len(data) < MIN_AUDIO_BYTES:
        raise InvalidAudioError("The recording is empty or incomplete.")
    if len(data) > max_bytes:
        raise AudioTooLargeError(f"The recording is too large (limit {max_bytes // 1_000_000} MB).")
    audio_format = detect_format(data)
    if audio_format is None:
        raise UnsupportedAudioError(f"This file is not a supported audio format. Please use {SUPPORTED}.")
    if audio_format is WAV:
        check_length(wav_duration(data), max_seconds)
    return audio_format


def check_length(seconds: float, max_seconds: float) -> None:
    if seconds > max_seconds:
        raise AudioTooLongError(f"The recording is too long (limit {max_seconds:.0f} seconds).")
