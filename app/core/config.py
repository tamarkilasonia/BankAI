"""Application configuration.

All configuration comes from environment variables (or a local ``.env`` file),
validated by Pydantic at startup. Nothing secret is hard-coded: secrets are
typed as ``SecretStr`` so they are masked in ``repr()``, logs and tracebacks.

Settings are added phase by phase; only settings that are actually used by the
code base live here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

Environment = Literal["development", "test", "ci", "production"]


class Settings(BaseSettings):
    """Typed, validated application settings.

    Environment variable names are the upper-case field names, e.g.
    ``DATABASE_URL`` -> ``database_url``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "BankAI"
    environment: Environment = "development"

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://bankai:bankai@localhost:5432/bankai"
    database_echo: bool = Field(default=False, description="Log every SQL statement (debug only).")

    # --- Mock data ---
    seed_random_seed: int = Field(
        default=42,
        description="Seed for the mock data generator. Same seed -> identical mock data.",
    )

    # --- MCP tool server ---
    mcp_server_url: str = Field(
        default="http://localhost:8001/mcp", description="Where the agent reaches the MCP server."
    )
    mcp_host: str = Field(default="127.0.0.1", description="Interface the MCP server listens on.")
    mcp_port: int = Field(default=8001, ge=1, le=65535)
    mcp_allowed_hosts: list[str] = Field(
        default=["localhost:*", "127.0.0.1:*"],
        description="Host headers the MCP server accepts (DNS-rebinding protection). JSON list in env.",
    )
    mcp_auth_secret: SecretStr | None = Field(
        default=None,
        description="Shared HMAC secret for agent -> MCP principal tokens. At least 32 characters.",
    )
    mcp_token_ttl_seconds: int = Field(default=60, ge=5, le=3600)
    tool_timeout_seconds: float = Field(default=10.0, gt=0, le=120)

    # --- LLM ---
    # "none": no LLM at all - rule-based routing and template answers (offline, free,
    # deterministic; used by tests and CI). "openai_compatible": any server speaking
    # the OpenAI chat API - OpenAI itself, or a local Ollama (LLM_BASE_URL below).
    llm_provider: Literal["none", "openai_compatible"] = "none"
    llm_base_url: str | None = Field(
        default=None,
        description="None = api.openai.com. Ollama: http://localhost:11434/v1",
    )
    llm_model: str = Field(default="qwen2.5:3b", description="Model name as the server knows it.")
    llm_api_key: SecretStr | None = Field(default=None, description="Required by OpenAI; ignored by Ollama.")
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=400, ge=16, le=4000)

    # --- RAG (knowledge base) ---
    # hashing: offline lexical embeddings (tests, CI). openai_compatible: e.g. Ollama's
    # nomic-embed-text or OpenAI's text-embedding-3-small (with dimensions=768).
    embedding_provider: Literal["hashing", "openai_compatible"] = "hashing"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str | None = Field(default=None, description="Ollama: http://localhost:11434/v1")
    embedding_api_key: SecretStr | None = None
    # nomic-embed-text was trained with task prefixes; other models should use "".
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    vector_store: Literal["pgvector", "memory"] = "pgvector"
    rag_top_k: int = Field(default=4, ge=1, le=8)
    rag_min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold; None = the embedder's calibrated default.",
    )
    knowledge_base_dir: Path = Path("knowledge_base")

    # --- Agent ---
    # hybrid: deterministic rules first, LLM only when rules are unsure.
    router_mode: Literal["hybrid", "rules", "llm"] = "hybrid"
    # llm: the LLM phrases the answer (checked against the facts); template: never.
    response_mode: Literal["llm", "template"] = "llm"
    max_message_chars: int = Field(default=1000, ge=10, le=20000)

    # --- API ---
    # Signs the (MOCK) login session tokens. Unset outside production: a random per-process
    # secret is used, so sessions end when the server restarts. Required in production.
    api_session_secret: SecretStr | None = None
    api_session_ttl_minutes: int = Field(default=60, ge=1, le=24 * 60)
    # Protects the evaluation endpoints (they cost compute and show every mock customer's
    # answers). Unset = the endpoints are disabled.
    operator_api_key: SecretStr | None = None
    # How the API's agent reaches the tools: in the same process, or the MCP server over HTTP.
    mcp_transport: Literal["in_process", "http"] = "in_process"
    conversation_ttl_minutes: int = Field(default=30, ge=1, le=24 * 60)
    cors_origins: list[str] = Field(default=[], description="Extra browser origins (JSON list in env).")

    # --- Voice ---
    # Speech-to-text. "none": voice input disabled. "faster_whisper": local Whisper (free,
    # offline after the model download). "openai_compatible": an OpenAI-style
    # /audio/transcriptions endpoint (OpenAI itself needs STT_API_KEY).
    stt_provider: Literal["none", "faster_whisper", "openai_compatible"] = "none"
    stt_model: str = Field(
        default="small", description="Whisper size (faster_whisper) or model name (OpenAI)."
    )
    stt_model_dir: Path = Path("models/whisper")
    stt_base_url: str | None = None
    stt_api_key: SecretStr | None = None
    stt_prompt: str = Field(
        default=(
            "A customer asks a banking assistant about lari (GEL), dollars (USD) and euros (EUR), "
            "accounts, cards and transfers."
        ),
        description="Domain vocabulary hint for the recognizer (fixed 'lari' heard as 'lorry'); '' = none.",
    )
    stt_min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this transcript confidence the assistant asks to repeat instead of acting.",
    )
    # Text-to-speech. "piper": local neural voices (one per language). "openai_compatible":
    # an OpenAI-style /audio/speech endpoint.
    tts_provider: Literal["none", "piper", "openai_compatible"] = "none"
    tts_voices_dir: Path = Path("models/piper")
    tts_voice_en: str = "en_US-lessac-medium"
    tts_voice_ka: str = "ka_GE-natia-medium"
    tts_model: str = Field(default="gpt-4o-mini-tts", description="openai_compatible only")
    tts_base_url: str | None = None
    tts_api_key: SecretStr | None = None
    voice_max_audio_bytes: int = Field(default=10_000_000, ge=1000, le=50_000_000)
    voice_max_audio_seconds: float = Field(default=60.0, gt=0, le=600)
    tts_max_chars: int = Field(default=1500, ge=10, le=10000)

    @field_validator(
        "mcp_auth_secret",
        "llm_api_key",
        "embedding_api_key",
        "rag_min_score",
        "stt_api_key",
        "tts_api_key",
        "api_session_secret",
        "operator_api_key",
        mode="before",
    )
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """``KEY=`` in a copied .env.example means "not configured", not "empty secret"."""
        return None if value == "" else value

    @field_validator("mcp_auth_secret", "api_session_secret", "operator_api_key")
    @classmethod
    def _secret_long_enough(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("secrets and keys must be at least 32 characters (HS256 key strength)")
        return value

    @property
    def database_url_safe(self) -> str:
        """Database URL with the password masked - safe to log."""
        return make_url(self.database_url).render_as_string(hide_password=True)

    def require_mcp_auth_secret(self) -> str:
        """The MCP secret, or a clear startup error. Never falls back to a default."""
        if self.mcp_auth_secret is None:
            raise ConfigurationError(
                "MCP_AUTH_SECRET is not set. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return self.mcp_auth_secret.get_secret_value()


class ConfigurationError(RuntimeError):
    """Required configuration is missing or invalid."""


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so the environment is parsed once. Tests that change environment
    variables must call ``get_settings.cache_clear()``.
    """
    return Settings()
