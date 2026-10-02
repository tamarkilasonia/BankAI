"""A voice turn end to end: audio -> STT -> agent -> TTS, with fake speech engines and the real agent."""

import pytest
from mcp.server.mcpserver import MCPServer

from app.agents.assistant import BankingAssistant
from app.agents.intents import Intent
from app.agents.nodes import AgentDeps
from app.mcp.client import MCPToolGateway
from app.models.enums import SafetyStatus
from app.voice.errors import AudioTooLongError, InvalidAudioError, SpeechUnavailableError, SynthesisError
from app.voice.pipeline import NOT_UNDERSTOOD, VoicePipeline
from app.voice.stt import Transcription
from tests.constants import ANCHOR, NINO, TEST_MCP_SECRET
from tests.fakes import FakeSTT, FakeTTS, ScriptedLLM, wav_bytes


def heard(
    text: str, *, language: str | None = "en", confidence: float = 0.8, duration: float = 2.0
) -> Transcription:
    return Transcription(text, language, 0.99, confidence, duration, "fake-stt", 5.0)


def pipeline(server: MCPServer, stt: FakeSTT, tts: FakeTTS | None = None, **options: object) -> VoicePipeline:
    gateway = MCPToolGateway(server, auth_secret=TEST_MCP_SECRET)
    assistant = BankingAssistant(AgentDeps(gateway=gateway, clock=lambda: ANCHOR))
    return VoicePipeline(assistant, stt, tts if tts is not None else FakeTTS(), **options)  # type: ignore[arg-type]


async def test_a_spoken_question_is_answered_in_text_and_speech(tool_server: MCPServer) -> None:
    tts = FakeTTS()
    turn = await pipeline(tool_server, FakeSTT(heard("What's my current balance?")), tts).converse(
        wav_bytes(), user_id=NINO
    )

    assert turn.understood and turn.reason is None
    assert turn.reply is not None and turn.reply.intent is Intent.ACCOUNT_BALANCE
    assert turn.response_text.startswith("Here are your current balances:")
    assert turn.speech is not None and turn.speech.content_type == "audio/wav"
    spoken, language = tts.spoken[0]
    assert language == "en" and "ending in 0001" in spoken and "****" not in spoken
    assert turn.errors == []


@pytest.mark.parametrize(
    ("transcription", "reason"),
    [
        (heard(""), "empty"),
        (heard(f"?! ... {chr(0x589)} {chr(0x589)}"), "empty"),  # measured: Georgian came back as punctuation
        (heard("Transfer 500 lorry to my mom", confidence=0.31), "low_confidence"),
        (heard("Ramdeni tanxa makvs", language="fa", confidence=0.9), "unsupported_language"),
    ],
)
async def test_unclear_speech_is_never_acted_on(
    tool_server: MCPServer, transcription: Transcription, reason: str
) -> None:
    turn = await pipeline(tool_server, FakeSTT(transcription)).converse(wav_bytes(), user_id=NINO)

    assert not turn.understood and turn.reason == reason
    assert turn.reply is None  # the agent was never called
    assert turn.response_text == NOT_UNDERSTOOD["en"]


async def test_unclear_georgian_gets_a_georgian_answer(tool_server: MCPServer) -> None:
    tts = FakeTTS()
    turn = await pipeline(
        tool_server, FakeSTT(heard("Ramdenitan Han makus", language="ka", confidence=0.46)), tts
    ).converse(wav_bytes(), user_id=NINO)

    assert turn.response_text == NOT_UNDERSTOOD["ka"]
    assert tts.spoken[0][1] == "ka"


async def test_the_language_hint_reaches_the_recognizer(tool_server: MCPServer) -> None:
    stt = FakeSTT(heard("რამდენი თანხა მაქვს?", language="ka"))
    turn = await pipeline(tool_server, stt).converse(wav_bytes(), user_id=NINO, language="ka")

    assert stt.calls == [{"bytes": len(wav_bytes()), "language": "ka"}]
    assert turn.reply is not None and turn.reply.language == "ka"


async def test_the_confidence_threshold_is_configurable(tool_server: MCPServer) -> None:
    strict = pipeline(tool_server, FakeSTT(heard("What's my balance?", confidence=0.8)), min_confidence=0.9)
    assert (await strict.converse(wav_bytes(), user_id=NINO)).reason == "low_confidence"


async def test_a_spoken_transfer_still_needs_a_confirmation_turn(tool_server: MCPServer) -> None:
    voice = pipeline(tool_server, FakeSTT(heard("Transfer 500 GEL to my mom"), heard("yes")))

    first = await voice.converse(wav_bytes(), user_id=NINO)
    assert first.reply is not None and first.reply.safety_status is SafetyStatus.CONFIRMATION_REQUIRED
    second = await voice.converse(wav_bytes(), user_id=NINO, pending=first.reply.pending_confirmation)

    assert second.reply is not None and second.reply.action_request is not None
    assert "no money was moved" in second.response_text


# --- failures --------------------------------------------------------------------------------------


async def test_tts_failure_keeps_the_text_answer(tool_server: MCPServer) -> None:
    tts = FakeTTS(error=SynthesisError("Speech synthesis failed."))
    turn = await pipeline(tool_server, FakeSTT(heard("What's my balance?")), tts).converse(
        wav_bytes(), user_id=NINO
    )

    assert turn.speech is None and turn.errors == ["tts_failed"]
    assert turn.response_text.startswith("Here are your current balances:")


async def test_without_tts_there_is_no_audio_and_no_error(tool_server: MCPServer) -> None:
    gateway = MCPToolGateway(tool_server, auth_secret=TEST_MCP_SECRET)
    voice = VoicePipeline(BankingAssistant(AgentDeps(gateway=gateway)), FakeSTT(heard("hi")), None)  # type: ignore[arg-type]
    turn = await voice.converse(wav_bytes(), user_id=NINO)
    assert (turn.speech, turn.errors) == (None, [])


async def test_a_language_without_a_voice_is_read_in_english(tool_server: MCPServer) -> None:
    tts = FakeTTS(languages=frozenset({"en"}))
    await pipeline(tool_server, FakeSTT(heard("რამდენი თანხა მაქვს?", language="ka")), tts).converse(
        wav_bytes(), user_id=NINO
    )
    assert tts.spoken[0][1] == "en"


async def test_bad_or_long_audio_is_rejected_before_or_after_decoding(tool_server: MCPServer) -> None:
    stt = FakeSTT(heard("a very long ramble", duration=120.0))
    voice = pipeline(tool_server, stt, max_audio_seconds=60.0)

    with pytest.raises(InvalidAudioError):
        await voice.converse(b"", user_id=NINO)
    with pytest.raises(AudioTooLongError):  # WAV: measured before transcription
        await voice.converse(wav_bytes(61.0), user_id=NINO)
    assert stt.calls == []
    with pytest.raises(AudioTooLongError):  # compressed: measured by the decoder
        await voice.converse(b"OggS" + b"\x00" * 100, user_id=NINO)


async def test_stt_unavailable_is_an_error_not_an_answer(tool_server: MCPServer) -> None:
    voice = pipeline(tool_server, FakeSTT(SpeechUnavailableError("Speech recognition is unreachable.")))
    with pytest.raises(SpeechUnavailableError):
        await voice.converse(wav_bytes(), user_id=NINO)


async def test_llm_answers_are_spoken_too(tool_server: MCPServer) -> None:
    gateway = MCPToolGateway(tool_server, auth_secret=TEST_MCP_SECRET)
    assistant = BankingAssistant(
        AgentDeps(
            gateway=gateway,
            llm=ScriptedLLM("Hello! How can I help with your banking today?"),
            clock=lambda: ANCHOR,
        )
    )
    tts = FakeTTS()
    turn = await VoicePipeline(assistant, FakeSTT(heard("hello")), tts).converse(wav_bytes(), user_id=NINO)  # type: ignore[arg-type]

    assert tts.spoken == [("Hello! How can I help with your banking today?", "en")]
    assert turn.reply is not None and turn.reply.response_source == "llm"
