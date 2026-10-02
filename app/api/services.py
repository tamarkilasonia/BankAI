"""Everything the API needs, built once at startup and closed at shutdown.

Route handlers get these through FastAPI dependencies (``app/api/dependencies.py``),
never through globals - tests build a ``Services`` with fakes and inject it.
"""

import asyncio
import secrets
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.assistant import BankingAssistant
from app.agents.factory import open_assistant
from app.api.conversations import ConversationStore
from app.core.config import ConfigurationError, Settings
from app.core.logging import get_logger
from app.database.session import get_sessionmaker
from app.voice.factory import create_stt, create_tts
from app.voice.pipeline import VoicePipeline
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech

logger = get_logger(__name__)


@dataclass
class Services:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession]
    assistant: BankingAssistant
    session_secret: str
    conversations: ConversationStore
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None
    evaluation_task: asyncio.Task[object] | None = None
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    @property
    def voice(self) -> VoicePipeline | None:
        if self.stt is None:
            return None
        return VoicePipeline(
            self.assistant,
            self.stt,
            self.tts,
            min_confidence=self.settings.stt_min_confidence,
            max_audio_bytes=self.settings.voice_max_audio_bytes,
            max_audio_seconds=self.settings.voice_max_audio_seconds,
            tts_max_chars=self.settings.tts_max_chars,
        )

    async def aclose(self) -> None:
        if self.evaluation_task is not None and not self.evaluation_task.done():
            self.evaluation_task.cancel()
        for provider in (self.stt, self.tts):
            if provider is not None:
                await provider.aclose()
        await self.exit_stack.aclose()


def session_secret(settings: Settings) -> str:
    if settings.api_session_secret is not None:
        return settings.api_session_secret.get_secret_value()
    if settings.environment == "production":
        raise ConfigurationError("API_SESSION_SECRET is required in production.")
    logger.warning("api.ephemeral_session_secret")  # sessions end when the process restarts
    return secrets.token_urlsafe(32)


async def build_services(settings: Settings) -> Services:
    sessions = get_sessionmaker()
    stack = AsyncExitStack()
    mcp_url = settings.mcp_server_url if settings.mcp_transport == "http" else None
    assistant = await stack.enter_async_context(open_assistant(settings, sessions, mcp_url=mcp_url))
    return Services(
        settings=settings,
        sessions=sessions,
        assistant=assistant,
        session_secret=session_secret(settings),
        conversations=ConversationStore(timedelta(minutes=settings.conversation_ttl_minutes)),
        stt=create_stt(settings),
        tts=create_tts(settings),
        exit_stack=stack,
    )
