"""Evaluation cases and results.

A case says *what* the right outcome is, never the number itself: "the
customer's restaurant spending this month" is an ``OracleQuery``, and the
oracle computes the value from the database when the case runs. So the
dataset cannot go stale when the mock data changes, and no expected number
is ever typed in by hand.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.intents import Intent
from app.models.enums import Channel


class EvalCategory(StrEnum):
    ACCOUNT = "ACCOUNT"
    TRANSACTION = "TRANSACTION"
    SPENDING = "SPENDING"
    CALCULATION = "CALCULATION"  # exchange rates and loan payments
    RAG = "RAG"
    GENERAL = "GENERAL"
    SAFETY = "SAFETY"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    AUTHORIZATION = "AUTHORIZATION"
    VOICE = "VOICE"  # transcript-style input: no punctuation, fillers, spoken numbers


class Behavior(StrEnum):
    """What the assistant did, read from its reply (see ``scoring.observed_behavior``)."""

    ANSWER = "answer"  # answered from tool data
    DOCUMENT_ANSWER = "document_answer"  # answered from the knowledge base, with sources
    NOT_ENOUGH_INFORMATION = "not_enough_information"  # the documents don't cover it
    CLARIFY = "clarify"  # asked for missing details instead of guessing
    RULE_VIOLATION = "rule_violation"  # explained a banking rule (e.g. loan limits)
    GENERAL = "general"  # greeting, capabilities, polite "I can't help with that topic"
    REFUSE = "refuse"
    CONFIRM = "confirm"  # high-risk action: asked for an explicit confirmation
    CONFIRMED_MOCK = "confirmed_mock"  # after "yes": a MOCK request was recorded
    CANCELLED = "cancelled"  # after "no" (or an expired confirmation)
    INPUT_ERROR = "input_error"  # empty or too long
    ERROR = "error"  # a tool failed unexpectedly


ANSWERING = frozenset(
    {Behavior.ANSWER, Behavior.DOCUMENT_ANSWER, Behavior.GENERAL, Behavior.CLARIFY, Behavior.RULE_VIOLATION}
)

OracleKind = Literal[
    "balances",
    "account_numbers",
    "transactions",
    "search",
    "spending_total",
    "top_category",
    "exchange",
    "loan",
]


class OracleQuery(BaseModel):
    """A question for the oracle, e.g. ``{"kind": "spending_total", "params": {"period": "this_month"}}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: OracleKind
    params: dict[str, Any] = Field(default_factory=dict)


class AnswerExpectation(BaseModel):
    """Characteristics the answer must have."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    oracle: OracleQuery | None = None  # values computed from the database must appear
    values: list[Decimal] = Field(default_factory=list)  # numbers from the documents ("3,000" == "3000.00")
    contains: list[str] = Field(default_factory=list)  # each must appear (case-insensitive)
    not_contains: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)  # relevance: at least one must appear
    sources: list[str] = Field(default_factory=list)  # knowledge-base files that must be cited


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    category: EvalCategory
    input: str
    customer: str = "CUST-0001"
    channel: Channel = Channel.TEXT
    history: list[str] = Field(default_factory=list)  # earlier turns of the same conversation
    expected_intent: Intent | None = None
    expected_tool: str | None = None  # None: no tool may be selected
    expected_arguments: dict[str, Any] = Field(default_factory=dict)  # must be a subset of the actual ones
    expected_behavior: Behavior
    expected_answer: AnswerExpectation = Field(default_factory=AnswerExpectation)
    tags: list[str] = Field(default_factory=list)
    note: str = ""


class OracleAnswer(BaseModel):
    """What the oracle says must appear in a correct answer."""

    values: list[Decimal] = Field(default_factory=list)  # numbers
    phrases: list[str] = Field(default_factory=list)  # words (category names, account numbers...)


class CaseResult(BaseModel):
    """One case after scoring. ``scores`` hold True/False, or None when a metric does not apply."""

    case_id: str
    category: EvalCategory
    input: str
    customer: str
    tags: list[str]
    passed: bool
    scores: dict[str, bool | None]
    failure_reasons: list[str]
    expected_behavior: Behavior
    actual_behavior: Behavior
    expected_intent: str | None
    actual_intent: str
    expected_tool: str | None
    actual_tool: str | None
    response: str
    response_source: str
    llm_calls: int
    latency_ms: float
