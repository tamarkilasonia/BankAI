"""One voice turn: audio in, the agent's answer as text and speech out.

    validate audio ─► STT ─► understood? ──yes──► agent (channel=voice) ─► answer ─► TTS
                                  └──no──► "please repeat or type" (the agent is NOT called) ─► TTS

"Understood" means: a non-empty transcript, confidence at or above
``STT_MIN_CONFIDENCE``, and a language the assistant supports. Acting on misheard
speech is how a voice assistant transfers money to the wrong person; asking again
is cheap. (High-risk actions additionally need an explicit "yes" turn - phase 5.)

Failures degrade, they do not crash: bad audio and an unavailable STT raise a
``VoiceError`` (the API turns it into a clear 4xx/503); a failing TTS still returns
the text answer, just without audio.
"""

import re
import uuid

from pydantic import BaseModel, ConfigDict

from app.agents.assistant import AssistantReply, BankingAssistant, PendingConfirmation, PreviousTurn
from app.core.logging import get_logger
from app.models.enums import Channel
from app.voice.audio import check_length, validate_audio
from app.voice.errors import VoiceError
from app.voice.speech import speakable
from app.voice.stt import SpeechToText, Transcription
from app.voice.tts import SpeechAudio, TextToSpeech

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = frozenset({"en", "ka"})
_WORD = re.compile(r"\w")
NOT_UNDERSTOOD = {
    "en": (
        "Sorry, I didn't catch that clearly, so I haven't done anything. "
        "Please say it again, or type your question."
    ),
    "ka": "ბოდიში, კარგად ვერ გავიგე, ამიტომ არაფერი გამიკეთებია. გთხოვთ, გაიმეოროთ ან დაწეროთ კითხვა.",
}


class VoiceTurn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transcription: Transcription
    understood: bool
    reason: str | None  # why the speech was not acted on: empty | low_confidence | unsupported_language
    reply: AssistantReply | None  # None when the agent was not called
    response_text: str
    speech: SpeechAudio | None  # None when TTS is disabled or failed
    errors: list[str]


class VoicePipeline:
    def __init__(
        self,
        assistant: BankingAssistant,
        stt: SpeechToText,
        tts: TextToSpeech | None,
        *,
        min_confidence: float = 0.5,
        max_audio_bytes: int = 10_000_000,
        max_audio_seconds: float = 60.0,
        tts_max_chars: int = 1500,
    ) -> None:
        self.assistant = assistant
        self.stt = stt
        self.tts = tts
        self.min_confidence = min_confidence
        self.max_audio_bytes = max_audio_bytes
        self.max_audio_seconds = max_audio_seconds
        self.tts_max_chars = tts_max_chars

    def why_not_understood(self, transcription: Transcription, language_hint: str | None) -> str | None:
        if not _WORD.search(transcription.text):  # nothing, or only punctuation (measured on Georgian)
            return "empty"
        if transcription.confidence < self.min_confidence:
            return "low_confidence"
        if (transcription.language or language_hint) not in SUPPORTED_LANGUAGES:
            return "unsupported_language"
        return None

    async def converse(
        self,
        audio: bytes,
        *,
        user_id: uuid.UUID,
        language: str | None = None,
        previous: PreviousTurn | None = None,
        pending: PendingConfirmation | None = None,
        request_id: str | None = None,
    ) -> VoiceTurn:
        validate_audio(audio, max_bytes=self.max_audio_bytes, max_seconds=self.max_audio_seconds)
        transcription = await self.stt.transcribe(audio, language=language)
        check_length(transcription.duration_s, self.max_audio_seconds)

        reason = self.why_not_understood(transcription, language)
        reply = None
        if reason is None:
            reply = await self.assistant.respond(
                user_id=user_id,
                message=transcription.text,
                channel=Channel.VOICE,
                previous=previous,
                pending=pending,
                request_id=request_id,
            )
            text, speak_in = reply.response, reply.language
        else:
            # Answer in the language the customer most likely spoke, even if the words were unclear.
            likely = [lang for lang in (language, transcription.language) if lang in SUPPORTED_LANGUAGES]
            speak_in = likely[0] if likely else "en"
            text = NOT_UNDERSTOOD[speak_in]
        logger.info(
            "voice.turn",
            extra={
                "stt_model": transcription.model,
                "stt_latency_ms": transcription.latency_ms,
                "detected_language": transcription.language,
                "confidence": transcription.confidence,
                "understood": reason is None,
                "reason": reason,
            },
        )
        speech, errors = await self.speak(text, language=speak_in)
        return VoiceTurn(
            transcription=transcription,
            understood=reason is None,
            reason=reason,
            reply=reply,
            response_text=text,
            speech=speech,
            errors=errors,
        )

    async def speak(self, text: str, *, language: str) -> tuple[SpeechAudio | None, list[str]]:
        """Synthesize ``text``; a failure costs the audio, never the answer."""
        if self.tts is None:
            return None, []
        if language not in self.tts.languages:
            language = "en"
        try:
            return await self.tts.synthesize(
                speakable(text, language=language, max_chars=self.tts_max_chars), language=language
            ), []
        except VoiceError as exc:
            logger.warning("voice.tts_failed", extra={"code": exc.code})
            return None, [exc.code]
