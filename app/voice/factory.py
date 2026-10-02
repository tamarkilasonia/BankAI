"""Speech providers from settings. ``none`` disables the direction (the API then says so)."""

from app.core.config import Settings
from app.voice.stt import FasterWhisperSTT, OpenAICompatibleSTT, SpeechToText
from app.voice.tts import OpenAICompatibleTTS, PiperTTS, TextToSpeech


def create_stt(settings: Settings) -> SpeechToText | None:
    if settings.stt_provider == "faster_whisper":
        return FasterWhisperSTT(
            settings.stt_model, model_dir=settings.stt_model_dir, prompt=settings.stt_prompt
        )
    if settings.stt_provider == "openai_compatible":
        return OpenAICompatibleSTT(
            model=settings.stt_model,
            base_url=settings.stt_base_url,
            api_key=settings.stt_api_key.get_secret_value() if settings.stt_api_key else None,
            prompt=settings.stt_prompt,
        )
    return None


def create_tts(settings: Settings) -> TextToSpeech | None:
    if settings.tts_provider == "piper":
        return PiperTTS(
            {"en": settings.tts_voice_en, "ka": settings.tts_voice_ka}, voices_dir=settings.tts_voices_dir
        )
    if settings.tts_provider == "openai_compatible":
        return OpenAICompatibleTTS(
            model=settings.tts_model,
            base_url=settings.tts_base_url,
            api_key=settings.tts_api_key.get_secret_value() if settings.tts_api_key else None,
        )
    return None
