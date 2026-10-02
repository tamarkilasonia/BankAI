"""Voice endpoints.

* ``POST /voice/transcribe`` - audio -> text (+ confidence, and whether the assistant would act on it).
* ``POST /voice/synthesize`` - text -> WAV.
* ``POST /voice/chat``       - the whole turn: audio -> transcript -> agent -> answer text + WAV.
  Low-confidence speech is answered with "please repeat" and never reaches the agent.
"""

import base64
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from app.api.conversations import log_turn
from app.api.dependencies import PrincipalDep, ServicesDep, VoiceDep
from app.api.routers.chat import chat_response
from app.api.schemas import (
    CONVERSATION_ID,
    ErrorResponse,
    SynthesizeRequest,
    TranscriptionResponse,
    VoiceChatResponse,
)
from app.core.logging import request_id_var
from app.models.enums import Channel
from app.voice.audio import check_length, validate_audio
from app.voice.errors import AudioTooLargeError, SpeechUnavailableError, TextTooLongError
from app.voice.pipeline import VoicePipeline
from app.voice.speech import speakable
from app.voice.stt import Transcription

router = APIRouter(prefix="/voice", tags=["voice"])

ERRORS = {
    code: {"model": ErrorResponse}
    for code in (400, 401, 413, 415, 422, 502, 503)
}  # fmt: skip
AudioFile = Annotated[UploadFile, File(description="WAV, MP3, OGG, WebM, MP4/M4A or FLAC")]
LanguageHint = Annotated[Literal["en", "ka"] | None, Form(description="Skip language detection")]


async def read_audio(upload: UploadFile, max_bytes: int) -> bytes:
    """Read at most one byte more than allowed, so a huge upload is never held in memory."""
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AudioTooLargeError(f"The recording is too large (limit {max_bytes // 1_000_000} MB).")
    return data


def transcription_out(
    transcription: Transcription, voice: VoicePipeline, hint: str | None
) -> TranscriptionResponse:
    reason = voice.why_not_understood(transcription, hint)
    return TranscriptionResponse(
        text=transcription.text,
        language=transcription.language,
        language_probability=transcription.language_probability,
        confidence=transcription.confidence,
        duration_s=transcription.duration_s,
        understood=reason is None,
        reason=reason,
        model=transcription.model,
    )


@router.post("/transcribe", responses=ERRORS)  # type: ignore[arg-type]
async def transcribe(
    audio: AudioFile, _: PrincipalDep, voice: VoiceDep, language: LanguageHint = None
) -> TranscriptionResponse:
    """Speech to text. ``understood`` tells whether the assistant would act on the transcript."""
    data = await read_audio(audio, voice.max_audio_bytes)
    validate_audio(data, max_bytes=voice.max_audio_bytes, max_seconds=voice.max_audio_seconds)
    transcription = await voice.stt.transcribe(data, language=language)
    check_length(transcription.duration_s, voice.max_audio_seconds)
    return transcription_out(transcription, voice, language)


@router.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "WAV audio"}} | ERRORS,  # type: ignore[operator]
)
async def synthesize(body: SynthesizeRequest, _: PrincipalDep, services: ServicesDep) -> Response:
    """Text to speech (WAV). The text is read the way the assistant speaks its answers."""
    if services.tts is None:
        raise SpeechUnavailableError("Voice output is not enabled on this server (TTS_PROVIDER).")
    if len(body.text) > services.settings.tts_max_chars:
        raise TextTooLongError(
            f"The text is too long to read aloud (limit {services.settings.tts_max_chars})."
        )
    speech = await services.tts.synthesize(
        speakable(body.text, language=body.language), language=body.language
    )
    return Response(content=speech.audio, media_type=speech.content_type)


@router.post("/chat", responses=ERRORS)  # type: ignore[arg-type]
async def voice_chat(
    audio: AudioFile,
    principal: PrincipalDep,
    services: ServicesDep,
    voice: VoiceDep,
    language: LanguageHint = None,
    conversation_id: Annotated[str | None, Form(pattern=CONVERSATION_ID)] = None,
) -> VoiceChatResponse:
    """A whole spoken turn: the transcript, the agent's answer as text, and the answer as speech."""
    data = await read_audio(audio, voice.max_audio_bytes)
    conversation_id, conversation = services.conversations.open(conversation_id, principal.user_id)
    request_id = request_id_var.get() or ""
    turn = await voice.converse(
        data,
        user_id=principal.user_id,
        language=language,
        previous=conversation.previous,
        pending=conversation.pending,
        request_id=request_id,
    )
    if turn.reply is not None:
        services.conversations.remember(conversation_id, turn.reply)
    await log_turn(
        services.sessions,
        conversation_id=conversation_id,
        user_id=principal.user_id,
        channel=Channel.VOICE,
        user_input=turn.transcription.text,
        reply=turn.reply,
        response_text=turn.response_text,
        request_id=request_id,
        error_code=f"not_understood:{turn.reason}" if turn.reason else None,
    )
    speech = turn.speech
    return VoiceChatResponse(
        conversation_id=conversation_id,
        request_id=request_id,
        transcription=transcription_out(turn.transcription, voice, language),
        response=turn.response_text,
        chat=chat_response(conversation_id, turn.reply) if turn.reply else None,
        audio_base64=base64.b64encode(speech.audio).decode() if speech else None,
        audio_content_type=speech.content_type if speech else None,
        errors=turn.errors,
    )
