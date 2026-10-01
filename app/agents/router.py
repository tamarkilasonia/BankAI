"""Hybrid intent routing: rules first, the LLM when the rules are unsure.

    message ──► RuleRouter ──┬─ unsafe / high-risk ───────────────► decided (LLM never overrides)
                             ├─ confident match (hybrid mode) ────► decided
                             └─ unsure / no match ──► LLM ──┬─ valid JSON ──► decided
                                                            └─ error ───────► rule guess, else "general"

Router modes (ROUTER_MODE): ``hybrid`` (default), ``rules`` (never call the LLM),
``llm`` (always ask the LLM, except for the safety patterns).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.agents.intents import INTENT_TOOL, Intent
from app.agents.prompts import router_messages
from app.agents.rules import RouteDecision, RuleRouter
from app.core.logging import get_logger
from app.core.time import BANK_TZ
from app.llm import LLMClient, LLMError, complete_structured
from app.mcp.schemas import TOOLS

logger = get_logger(__name__)

ACCEPT_RULES_AT = 0.8
_SAFETY_INTENTS = frozenset({Intent.UNSAFE, Intent.HIGH_RISK_ACTION})


class LLMRouting(BaseModel):
    """The JSON the router asks the LLM for. Validated like any other untrusted input."""

    intent: Intent
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass
class RoutingOutcome:
    decision: RouteDecision
    llm_calls: int = 0
    llm_latency_ms: float | None = None
    errors: list[str] = field(default_factory=list)


class IntentRouter:
    def __init__(
        self, *, rules: RuleRouter, llm: LLMClient | None, mode: str, clock: Callable[[], datetime]
    ) -> None:
        self._rules = rules
        self._llm = llm
        self._mode = mode
        self._clock = clock

    async def route(self, message: str, *, previous: RouteDecision | None = None) -> RoutingOutcome:
        rule = self._rules.route(message, previous=previous)
        if rule is not None and rule.intent in _SAFETY_INTENTS:
            return RoutingOutcome(rule)

        use_llm = self._llm is not None and self._mode != "rules"
        # In hybrid mode a confident AND complete rule decision is final. Incomplete ones
        # ("car loan over four years" without a parsed term) go to the LLM, which may fill the gaps.
        trust_rule = rule is not None and (
            not use_llm
            or (self._mode == "hybrid" and rule.confidence >= ACCEPT_RULES_AT and not rule.missing)
        )
        if rule is not None and trust_rule:
            return RoutingOutcome(rule)

        errors: list[str] = []
        llm_calls = 0
        if use_llm and self._llm is not None:
            llm_calls = 1
            try:
                messages = router_messages(
                    message,
                    today=self._clock().astimezone(BANK_TZ).date(),
                    previous_intent=previous.intent.value if previous else None,
                )
                routing, completion = await complete_structured(
                    self._llm, messages, LLMRouting, max_tokens=200
                )
            except LLMError as exc:
                logger.warning("router.llm_failed", extra={"code": exc.code})
                errors.append(exc.code)
            else:
                decision = RouteDecision(
                    intent=routing.intent,
                    arguments=_known_arguments(routing.intent, routing.arguments),
                    confidence=routing.confidence,
                    source="llm",
                )
                return RoutingOutcome(decision, llm_calls=1, llm_latency_ms=completion.latency_ms)

        if rule is not None:
            return RoutingOutcome(rule, llm_calls=llm_calls, errors=errors)
        return RoutingOutcome(
            RouteDecision(Intent.GENERAL, confidence=0.0, source="fallback"),
            llm_calls=llm_calls,
            errors=errors,
        )


def _known_arguments(intent: Intent, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep only argument names the intent's tool declares (and drop nulls) - the rest is noise."""
    tool = INTENT_TOOL.get(intent)
    if tool is None:
        return {}
    fields = TOOLS[tool].input_model.model_fields
    return {key: value for key, value in arguments.items() if key in fields and value is not None}
