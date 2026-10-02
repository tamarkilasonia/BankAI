"""Request and response models of the HTTP API (they also generate the OpenAPI docs at /docs)."""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.enums import SafetyStatus, TransactionCategory, TransactionDirection
from app.safety.actions import ActionKind, ActionRequest
from app.services.periods import Period
from app.services.schemas import AccountDetails

CONVERSATION_ID = r"^[a-f0-9]{32}$"


class ErrorDetail(BaseModel):
    code: str = Field(examples=["access_denied"])
    message: str
    request_id: str | None
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# --- auth -------------------------------------------------------------------------------------------


class LoginRequest(BaseModel):
    customer_number: str = Field(pattern=r"^[Cc][Uu][Ss][Tt]-\d{4}$", examples=["CUST-0001"])


class CustomerProfile(BaseModel):
    user_id: uuid.UUID
    customer_number: str
    full_name: str
    preferred_language: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - the OAuth token type, not a secret
    expires_in: int = Field(description="Seconds until the session ends.")
    customer: CustomerProfile
    notice: str = "MOCK login for a prototype: no password, no real customer."


# --- chat ---------------------------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(max_length=10_000, examples=["What's my current balance?"])
    conversation_id: str | None = Field(
        default=None, pattern=CONVERSATION_ID, description="From the previous reply; omit to start a new one."
    )


class PendingActionOut(BaseModel):
    kind: ActionKind
    summary: str
    expires_at: datetime
    instructions: str = 'Reply "yes" to confirm or "no" to cancel. Nothing is ever executed: MOCK data.'


class ChatTrace(BaseModel):
    """How the agent got to its answer - shown in the demo UI, useful for debugging."""

    intent: str
    intent_source: str
    intent_confidence: float
    tool: str | None
    tool_arguments: dict[str, Any] | None
    tool_error: str | None
    response_source: str
    safety_flags: list[str]
    llm_calls: int
    errors: list[str]
    timings_ms: dict[str, float]


class ChatResponse(BaseModel):
    conversation_id: str
    request_id: str
    response: str
    language: str
    safety_status: SafetyStatus | None
    pending_confirmation: PendingActionOut | None
    action_request: ActionRequest | None
    latency_ms: float
    trace: ChatTrace


# --- voice --------------------------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None
    language_probability: float | None
    confidence: float
    duration_s: float
    understood: bool = Field(description="Would the assistant act on this transcript?")
    reason: str | None = Field(description="Why not: empty | low_confidence | unsupported_language")
    model: str


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    language: Literal["en", "ka"] = "en"


class VoiceChatResponse(BaseModel):
    conversation_id: str
    request_id: str
    transcription: TranscriptionResponse
    response: str
    chat: ChatResponse | None = Field(
        description="The agent's answer; None when the speech was not understood"
    )
    audio_base64: str | None = Field(description="The spoken answer (WAV), base64-encoded")
    audio_content_type: str | None
    errors: list[str]


# --- banking data ------------------------------------------------------------------------------------


class AccountsResponse(BaseModel):
    customer_number: str
    full_name: str
    accounts: list[AccountDetails]
    totals_by_currency: dict[str, Decimal]
    notice: str = "MOCK data"


class TransactionsQuery(BaseModel):
    period: Period = Period.LAST_30_DAYS
    start_date: date | None = None
    end_date: date | None = None
    category: TransactionCategory | None = None
    direction: TransactionDirection | None = None
    limit: int = Field(default=50, ge=1, le=100)


# --- evaluations --------------------------------------------------------------------------------------


class EvaluationRunRequest(BaseModel):
    categories: list[str] = Field(default=[], examples=[["SAFETY", "PROMPT_INJECTION"]])
    tags: list[str] = Field(default=[], examples=[["holdout"]])
    limit: int | None = Field(default=None, ge=1)


class EvaluationRunStarted(BaseModel):
    run_id: uuid.UUID
    status: str
    cases: int


class EvaluationRunOut(BaseModel):
    id: uuid.UUID
    status: str
    llm_provider: str
    model_name: str | None
    dataset_name: str
    dataset_version: str
    total_cases: int
    passed_cases: int
    started_at: datetime
    finished_at: datetime | None
    summary: dict[str, Any]


class EvaluationResultOut(BaseModel):
    case_id: str
    category: str
    input_text: str
    expected_behavior: str
    actual_behavior: str | None
    expected_intent: str | None
    actual_intent: str | None
    expected_tool: str | None
    actual_tool: str | None
    passed: bool
    scores: dict[str, Any]
    failure_reasons: list[str]
    response_text: str | None
    latency_ms: float | None


class EvaluationRunDetail(EvaluationRunOut):
    results: list[EvaluationResultOut]


# --- system -------------------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    environment: str
    database: Literal["ok", "unavailable"]
    components: dict[str, str] = Field(
        default={}, description="Configured providers (llm, embeddings, stt, tts, tools)"
    )
