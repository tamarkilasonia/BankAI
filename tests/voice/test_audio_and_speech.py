"""Audio checks (format by content, size, length, damage) and the speakable form of an answer."""

import pytest

from app.voice.audio import FORMATS, detect_format, validate_audio, wav_duration
from app.voice.errors import AudioTooLargeError, AudioTooLongError, InvalidAudioError, UnsupportedAudioError
from app.voice.speech import REST_ON_SCREEN, speakable
from tests.fakes import wav_bytes

LIMITS = {"max_bytes": 1_000_000, "max_seconds": 10.0}
PADDING = b"\x00" * 64


@pytest.mark.parametrize(
    ("header", "name"),
    [
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", "wav"),
        (b"ID3\x04\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x64", "mp3"),
        (b"OggS\x00\x02", "ogg"),
        (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81", "webm"),
        (b"\x00\x00\x00\x20ftypM4A ", "mp4"),
        (b"fLaC\x00\x00\x00\x22", "flac"),
    ],
)
def test_formats_are_recognized_by_content(header: bytes, name: str) -> None:
    assert detect_format(header + PADDING) == FORMATS[name]


def test_compressed_formats_pass_validation_and_wav_is_measured() -> None:
    assert validate_audio(b"OggS" + PADDING, **LIMITS).name == "ogg"
    assert validate_audio(wav_bytes(2.0), **LIMITS).content_type == "audio/wav"
    assert wav_duration(wav_bytes(2.5)) == 2.5


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (b"", InvalidAudioError, "empty"),
        (b"RIFF", InvalidAudioError, "empty"),
        (b"%PDF-1.7" + PADDING, UnsupportedAudioError, "not a supported audio format"),
        (b"\x89PNG\r\n\x1a\n" + PADDING, UnsupportedAudioError, "not a supported audio format"),
        (b"What's my balance?" * 4, UnsupportedAudioError, "not a supported audio format"),
        (b"RIFF\x24\x00\x00\x00WAVE" + b"garbage" * 10, InvalidAudioError, "damaged"),
        (wav_bytes(0.0), InvalidAudioError, "empty"),
        (wav_bytes(11.0), AudioTooLongError, "too long"),
        (b"OggS" + b"\x00" * 1_000_001, AudioTooLargeError, "too large"),
    ],
)
def test_bad_audio_is_rejected_with_a_clear_message(
    data: bytes, error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as caught:
        validate_audio(data, **LIMITS)
    assert message in str(caught.value)


def test_the_file_name_and_content_type_play_no_role() -> None:
    """A text file renamed to .wav is still a text file: only the bytes are checked."""
    assert detect_format(b"hello, this is not audio" * 3) is None


# --- what the voice reads ------------------------------------------------------------------------


def test_lists_masks_and_sources_are_made_speakable() -> None:
    answer = (
        "Here are your current balances:\n"
        "- Checking GEL (****0001): 8,687.37 GEL\n"
        "- Savings USD (****0003): 3,069.16 USD\n"
        "2026-09-27 | FuelPoint | -9.94 GEL | transportation\n\n"
        "Sources: Fee schedule: Card fees (fees.md)"
    )

    assert speakable(answer) == (
        "Here are your current balances: Checking GEL (ending in 0001): 8,687.37 GEL. "
        "Savings USD (ending in 0003): 3,069.16 USD. 2026-09-27, FuelPoint, -9.94 GEL, transportation."
    )


@pytest.mark.parametrize("language", ["en", "ka"])
def test_long_answers_are_cut_at_a_sentence_and_point_to_the_screen(language: str) -> None:
    answer = " ".join(f"Sentence number {i} is here." for i in range(100))

    spoken = speakable(answer, language=language, max_chars=200)

    assert len(spoken) <= 200
    assert spoken.endswith(f"is here. {REST_ON_SCREEN[language]}")


def test_a_text_without_sentence_breaks_is_still_cut() -> None:
    spoken = speakable("x" * 500, max_chars=100)
    assert len(spoken) <= 100 and spoken.endswith(REST_ON_SCREEN["en"])
