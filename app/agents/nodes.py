"""The LangGraph nodes - each one a small, testable step of the agent.

Every node receives the whole ``AgentState`` and returns only the keys it
changes. Nodes never raise for expected problems (bad input, tool errors, LLM
failures): they record them in the state and the graph routes to a safe answer.
The one deliberate exception is ``confirm_action``, which *pauses* the graph
(LangGraph ``interrupt``) until the customer answers "yes" or "no".
"""

import json
import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from app.agents import messages
from app.agents.grounding import missing_numbers, unsupported_numbers, unsupported_terms
from app.agents.intents import INTENT_TOOL, Intent
from app.agents.prompts import general_messages, knowledge_messages, responder_messages
from app.agents.rendering import render
from app.agents.router import IntentRouter
from app.agents.rules import RouteDecision, RuleRouter
from app.agents.state import AgentState
from app.core.logging import get_logger
from app.core.time import utcnow
from app.llm import ChatMessage, LLMClient, LLMError
from app.mcp.client import MCPToolGateway, ToolCallError
from app.mcp.schemas import TOOLS
from app.models.enums import SafetyStatus
from app.rag.schemas import PolicySearchResult
from app.safety import actions
from app.safety.actions import (
    CONFIRMATION_TTL_SECONDS,
    ActionKind,
    ActionRequest,
    ConfirmationReply,
    ProposedAction,
    asks_about_policy,
    classify_confirmation,
    describe_action,
)
from app.safety.input_guard import GuardAction, check_message, scan_untrusted
from app.safety.output_guard import OutputGuard
from app.safety.policy import refusal_for
from app.services.authorization import Principal

logger = get_logger(__name__)

_GEORGIAN = re.compile(f"[{chr(0x10A0)}-{chr(0x10FF)}]")  # Georgian script block
_PERIOD_KEYS = ("period", "start_date", "end_date")
# Arguments that point at a specific record. The LLM may only use values the customer typed.
_IDENTIFIER_ARGUMENTS = ("account_id",)
_ACTION_KINDS = {kind.value: kind for kind in ActionKind}

Update = dict[str, Any]


@dataclass(frozen=True)
class AgentConfig:
    router_mode: str = "hybrid"  # hybrid | rules | llm
    response_mode: str = "llm"  # llm | template
    max_message_chars: int = 1000


@dataclass(frozen=True)
class AgentDeps:
    gateway: MCPToolGateway
    llm: LLMClient | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    clock: Callable[[], datetime] = field(default=utcnow)
    output_guard: OutputGuard = field(default_factory=OutputGuard)


def clean_message(text: str) -> str:
    """Drop control characters (except newlines/tabs) and collapse whitespace."""
    kept = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return re.sub(r"[ \t]+", " ", kept).strip()


def prepare_arguments(tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Make proposed tool arguments valid, or say what is missing.

    Returns ``(arguments, missing, dropped)``. Invalid values (from rules or the
    LLM) are dropped one field at a time, so one bad value does not throw away
    the rest. Required fields that are absent are reported as ``missing``.
    """
    model = TOOLS[tool].input_model
    args = {k: v for k, v in arguments.items() if k in model.model_fields and v is not None}
    dropped: list[str] = []
    for _ in range(len(model.model_fields) + 1):
        try:
            model.model_validate(args)
            return args, [], dropped
        except ValidationError as exc:
            errors = exc.errors()
        missing = [str(e["loc"][0]) for e in errors if e["type"] == "missing" and e["loc"]]
        invalid = {str(e["loc"][0]) for e in errors if e["type"] != "missing" and e["loc"]}
        if missing and not invalid:
            return args, missing, dropped
        if not invalid:  # a cross-field rule failed (e.g. custom period without dates)
            invalid = {k for k in _PERIOD_KEYS if k in args}
            if not invalid:
                return args, ["details"], dropped
        for key in invalid:
            args.pop(key, None)
            dropped.append(key)
    return args, ["details"], dropped  # pragma: no cover - loop always returns


def unseen_identifiers(arguments: dict[str, Any], message: str) -> list[str]:
    """Identifier arguments whose value the customer never typed.

    An LLM has no legitimate way to know an account ID the customer did not
    mention, so such a value is either invented or planted by an injection. It is
    dropped before the tool call (authorization would refuse a foreign one anyway).
    """
    text = message.lower()
    return [
        key for key in _IDENTIFIER_ARGUMENTS if key in arguments and str(arguments[key]).lower() not in text
    ]


def with_sources(response: str, response_source: str, state: AgentState) -> str:
    """Append where a knowledge answer came from - added by code, so it cannot be invented.

    The template answer quotes only the best passage, so it cites only that one; an
    LLM answer was written from all retrieved passages, so it cites them all.
    """
    sources = state.get("sources") or []
    if not sources or messages.NO_INFORMATION in response:
        return response
    cited = sources if response_source == "llm" else sources[:1]
    return f"{response}\n\nSources: " + "; ".join(cited)


def screen_passages(result: PolicySearchResult) -> tuple[PolicySearchResult, int]:
    """Drop retrieved passages that contain instructions aimed at the assistant.

    Indirect prompt injection: a poisoned document must not get a chance to steer
    the LLM. If nothing clean is left, the question counts as not answerable.
    """
    kept = [
        p for p in result.passages if scan_untrusted(f"{p.section}\n{p.text}").action is GuardAction.ALLOW
    ]
    dropped = len(result.passages) - len(kept)
    if not dropped:
        return result, 0
    logger.warning(
        "safety.passages_dropped",
        extra={
            "dropped": dropped,
            "sources": sorted({p.source for p in result.passages} - {p.source for p in kept}),
        },
    )
    screened = result.model_copy(
        update={
            "passages": kept,
            "sufficient": result.sufficient and bool(kept),
            "reason": result.reason if kept else "unsafe_content",
        }
    )
    return screened, dropped


def _selected_tool(state: AgentState) -> str:
    tool = state.get("selected_tool")
    if tool is None:  # pragma: no cover - the graph only routes here with a tool selected
        raise RuntimeError("no tool selected")
    return tool


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.router = IntentRouter(
            rules=RuleRouter(deps.clock), llm=deps.llm, mode=deps.config.router_mode, clock=deps.clock
        )

    @property
    def _llm_answers(self) -> bool:
        return self.deps.llm is not None and self.deps.config.response_mode == "llm"

    # --- 1. input -----------------------------------------------------------------------

    async def validate_input(self, state: AgentState) -> Update:
        text = clean_message(state.get("message") or "")
        limit = self.deps.config.max_message_chars
        if not text:
            return {"input_error": "empty", "response": messages.EMPTY_INPUT}
        if len(text) > limit:
            return {"input_error": "too_long", "response": messages.TOO_LONG.format(limit=limit)}
        return {"message": text, "input_error": None, "language": "ka" if _GEORGIAN.search(text) else "en"}

    @staticmethod
    def after_validation(state: AgentState) -> str:
        return "invalid" if state.get("input_error") else "valid"

    async def input_guard(self, state: AgentState) -> Update:
        """Apply the safety policy before any LLM sees the message."""
        verdict = check_message(state["message"])
        if verdict.action is GuardAction.ALLOW:
            return {}
        decided: Update = {"intent_source": "safety", "intent_confidence": 1.0, "safety_flags": verdict.flags}
        if verdict.action is GuardAction.CONFIRM and verdict.action_kind is not None:
            return decided | {
                "intent": Intent.HIGH_RISK_ACTION.value,
                "route_reason": verdict.action_kind.value,
            }
        category = verdict.category.value if verdict.category else ""
        logger.warning(
            "safety.input_refused", extra={"category": category, "rule": verdict.rule, "view": verdict.view}
        )
        return decided | {
            "intent": Intent.UNSAFE.value,
            "route_reason": category,
            "safety_rule": verdict.rule,
        }

    @staticmethod
    def after_input_guard(state: AgentState) -> str:
        if state.get("intent_source") != "safety":
            return "allow"
        return "confirm" if state.get("intent") == Intent.HIGH_RISK_ACTION.value else "refuse"

    # --- 2. routing ---------------------------------------------------------------------

    async def classify_intent(self, state: AgentState) -> Update:
        previous = None
        if previous_intent := state.get("previous_intent"):
            previous = RouteDecision(Intent(previous_intent), dict(state.get("previous_arguments") or {}))
        outcome = await self.router.route(state["message"], previous=previous)
        decision = outcome.decision
        if decision.intent is Intent.HIGH_RISK_ACTION and asks_about_policy(state["message"]):
            # Same principle as the input guard: "Can I take money out of my deposit early?"
            # asks about a policy; it is not a request to move money.
            decision = replace(decision, intent=Intent.BANKING_KNOWLEDGE, reason="question_about_action")

        update: Update = {
            "intent": decision.intent.value,
            "intent_source": decision.source,
            "intent_confidence": decision.confidence,
            "route_reason": decision.reason,
            "llm_calls": outcome.llm_calls,
            "errors": list(outcome.errors),
            "safety_flags": [],
            "selected_tool": None,
            "tool_arguments": None,
            "missing_arguments": [],
        }
        if outcome.llm_latency_ms is not None:
            update["llm_latency_ms"] = outcome.llm_latency_ms
        if tool := INTENT_TOOL.get(decision.intent):
            proposed = dict(decision.arguments)
            if tool == "search_banking_policy":
                # Search with the customer's own words, not an LLM paraphrase of them.
                proposed = {"query": state["message"]}
            if decision.source == "llm":
                for key in unseen_identifiers(proposed, state["message"]):
                    proposed.pop(key)
                    update["safety_flags"] += [f"unseen_argument:{key}"]
            args, missing, dropped = prepare_arguments(tool, proposed)
            update |= {
                "selected_tool": tool,
                "tool_arguments": args,
                "missing_arguments": list(decision.missing) or missing,
            }
            update["errors"] += [f"dropped_argument:{key}" for key in dropped]
        if decision.intent is Intent.UNSAFE:
            update["safety_flags"] += [decision.reason or "llm_flagged_unsafe"]
        elif decision.intent is Intent.HIGH_RISK_ACTION:
            update["safety_flags"] += ["high_risk_action"]
        return update

    @staticmethod
    def after_classification(state: AgentState) -> str:
        intent = Intent(state["intent"])
        if intent in INTENT_TOOL:
            return "clarify" if state.get("missing_arguments") else "call_tool"
        return {
            Intent.GENERAL: "general_response",
            Intent.HIGH_RISK_ACTION: "high_risk_action",
            Intent.UNSAFE: "refuse",
        }[intent]

    # --- 3. tools -----------------------------------------------------------------------

    async def call_tool(self, state: AgentState) -> Update:
        tool = _selected_tool(state)
        principal = Principal(uuid.UUID(state["user_id"]))
        try:
            result = await self.deps.gateway.call(
                tool, state.get("tool_arguments") or {}, principal=principal
            )
        except ToolCallError as exc:
            return {"tool_error": {"code": exc.code, "message": exc.message}, "errors": [f"tool:{exc.code}"]}
        return {
            "tool_result": result.output.model_dump(mode="json"),
            "tool_arguments": result.arguments,
            "tool_latency_ms": result.duration_ms,
            "tool_error": None,
        }

    async def validate_tool_result(self, state: AgentState) -> Update:
        """Re-check the result's shape and turn it into facts + a template answer."""
        if state.get("tool_error"):
            return {}
        tool = _selected_tool(state)
        try:
            output = TOOLS[tool].output_model.model_validate(state.get("tool_result"))
        except ValidationError:
            return {
                "tool_error": {"code": "invalid_tool_output", "message": ""},
                "errors": ["invalid_tool_output"],
            }
        update: Update = {}
        if isinstance(output, PolicySearchResult):
            output, dropped = screen_passages(output)
            if dropped:
                update |= {
                    "tool_result": output.model_dump(mode="json"),
                    "safety_flags": ["indirect_injection_filtered"],
                }
        rendered = render(output)
        return update | {
            "facts": rendered.facts,
            "template_answer": rendered.answer,
            "key_numbers": [str(n) for n in rendered.key_numbers],
            "llm_may_rephrase": rendered.llm_may_rephrase,
            "prompt_kind": rendered.prompt_kind,
            "sources": list(rendered.sources),
            "disclaimer": rendered.disclaimer,
        }

    # --- 4. answers ---------------------------------------------------------------------

    async def generate_response(self, state: AgentState) -> Update:
        if error := state.get("tool_error"):
            return self._tool_error_answer(error)
        template = state["template_answer"]
        facts = state.get("facts", [])
        update: Update
        if not self._llm_answers or not state.get("llm_may_rephrase", True):
            update = {"response": template, "response_source": "template"}
        elif any(scan_untrusted(fact).action is not GuardAction.ALLOW for fact in facts):
            # Data from a tool (e.g. a payment description) contains instructions: the LLM never sees it.
            logger.warning("safety.tool_data_injection", extra={"tool": state.get("selected_tool")})
            update = {
                "response": template,
                "response_source": "template",
                "safety_flags": ["indirect_injection_in_tool_data"],
            }
        else:
            language = state.get("language", "en")
            if state.get("prompt_kind") == "passages":
                prompt = knowledge_messages(state["message"], facts, language)
            else:
                prompt = responder_messages(state["message"], facts, language)
            required = [Decimal(n) for n in state.get("key_numbers", [])]
            update = await self._llm_answer(
                prompt, fallback=template, sources=[*facts, state["message"]], required=required
            )
            disclaimer = state.get("disclaimer")
            if update["response_source"] == "llm" and disclaimer and disclaimer not in update["response"]:
                # Like sources: a required disclaimer is added by code, so the LLM cannot drop it.
                update["response"] = f"{update['response']} {disclaimer}"
        return update | {
            "response": with_sources(update["response"], update["response_source"], state),
            "fallback_response": with_sources(template, "template", state),
        }

    async def general_response(self, state: AgentState) -> Update:
        if state.get("intent_source") == "fallback":
            return {"response": messages.NOT_UNDERSTOOD, "response_source": "template"}
        if not self._llm_answers:
            return {"response": messages.GREETING, "response_source": "template"}
        prompt = general_messages(state["message"], state.get("language", "en"))
        update = await self._llm_answer(
            prompt, fallback=messages.GREETING, sources=[state["message"], messages.CAPABILITIES]
        )
        return update | {"fallback_response": messages.GREETING}

    async def clarify(self, state: AgentState) -> Update:
        text = messages.CLARIFICATIONS.get(state.get("selected_tool") or "", messages.DEFAULT_CLARIFICATION)
        return {"response": text, "response_source": "template"}

    async def refuse(self, state: AgentState) -> Update:
        return {
            "response": refusal_for(state.get("route_reason")),
            "response_source": "template",
            "safety_status": SafetyStatus.REFUSED.value,
        }

    # --- 5. high-risk actions: human in the loop ----------------------------------------------

    async def high_risk_action(self, state: AgentState) -> Update:
        """Work out what the customer asked for and ask them to confirm it.

        This runs *before* the pause, so its result (including the expiry time) is
        saved in the checkpoint and is identical when the graph resumes.
        """
        action = describe_action(state["message"], _ACTION_KINDS.get(state.get("route_reason") or ""))
        if not action.needs_confirmation:
            unclear = action.kind is ActionKind.OTHER  # only the LLM saw an action; we can't name it
            return {
                "response": actions.UNCLEAR_ACTION if unclear else actions.CREDENTIALS_NEVER_IN_CHAT,
                "response_source": "template",
                "safety_status": SafetyStatus.REFUSED.value,
                "safety_flags": ["unclear_action"] if unclear else [],
                "pending_action": None,
            }
        expires_at = self.deps.clock() + timedelta(seconds=CONFIRMATION_TTL_SECONDS)
        action = action.model_copy(update={"expires_at": expires_at})
        return {
            "pending_action": action.model_dump(mode="json"),
            "response": actions.confirmation_prompt(action),
            "response_source": "template",
            "safety_status": SafetyStatus.CONFIRMATION_REQUIRED.value,
        }

    @staticmethod
    def after_high_risk_action(state: AgentState) -> str:
        return "confirm" if state.get("pending_action") else "done"

    async def confirm_action(self, state: AgentState) -> Update:
        """Pause until the customer answers; then record a MOCK request or cancel.

        ``interrupt`` stops the graph here and saves its state. ``BankingAssistant``
        resumes it with the customer's next message; LangGraph then re-runs this node
        from the top, and ``interrupt`` returns that message instead of pausing.
        """
        action = ProposedAction.model_validate(state["pending_action"])
        answer = interrupt({"type": "confirmation_required", "action": state["pending_action"]})

        decision = self._confirmation_decision(state, action, answer)
        update: Update = {
            "confirmation": decision,
            "pending_action": None,
            "response_source": "template",
            "safety_status": SafetyStatus.ALLOWED.value,
            "safety_flags": [f"confirmation:{decision}"],
        }
        if decision == "confirmed":
            request = ActionRequest(
                reference=f"MOCK-{uuid.uuid4().hex[:10].upper()}",
                kind=action.kind,
                amount=action.amount,
                currency=action.currency,
                created_at=self.deps.clock(),
            )
            logger.info(
                "safety.mock_action_recorded",
                extra={"reference": request.reference, "kind": request.kind, "currency": request.currency},
            )
            return update | {
                "response": actions.confirmed_text(action, request),
                "action_request": request.model_dump(mode="json"),
            }
        texts = {
            "declined": actions.DECLINED,
            "expired": actions.EXPIRED,
            "superseded": actions.SUPERSEDED_NOTICE,
            "rejected": actions.REJECTED,
        }
        if decision == "rejected":
            update["safety_status"] = SafetyStatus.REFUSED.value
        return update | {"response": texts[decision]}

    def _confirmation_decision(self, state: AgentState, action: ProposedAction, answer: object) -> str:
        if not isinstance(answer, dict) or answer.get("user_id") != state["user_id"]:
            logger.warning("safety.confirmation_rejected")
            return "rejected"  # only the customer who asked may confirm
        reply = classify_confirmation(str(answer.get("message", "")))
        if reply is ConfirmationReply.OTHER:
            return "superseded"  # they asked something else: cancel, then answer that instead
        if action.expires_at is not None and self.deps.clock() > action.expires_at:
            return "expired"
        return "confirmed" if reply is ConfirmationReply.CONFIRM else "declined"

    # --- 6. output ------------------------------------------------------------------------

    async def output_guard(self, state: AgentState) -> Update:
        """Last check on the final answer: leaks, foreign identifiers, claims of actions."""
        response = state.get("response") or ""
        # Identifiers in this turn's own data (or typed by the customer) may be shown.
        allowed = [state.get("message") or "", json.dumps(state.get("tool_result") or {}, default=str)]
        findings = self.deps.output_guard.check(response, allowed=allowed)
        if not findings:
            return {}
        kinds = sorted({f.kind for f in findings})
        logger.warning(
            "safety.output_blocked", extra={"findings": [f"{f.kind}:{f.detail}" for f in findings]}
        )
        flags = [f"output_blocked:{kind}" for kind in kinds]
        fallback = state.get("fallback_response")
        if (
            state.get("response_source") == "llm"
            and fallback
            and not self.deps.output_guard.check(fallback, allowed=allowed)
        ):
            return {
                "response": fallback,
                "response_source": "template",
                "safety_flags": flags,
                "errors": ["output_guard_fallback"],
            }
        return {
            "response": messages.OUTPUT_BLOCKED,
            "response_source": "template",
            "safety_status": SafetyStatus.REFUSED.value,
            "safety_flags": flags,
            "errors": ["output_blocked"],
        }

    async def finalize(self, state: AgentState) -> Update:
        return {"safety_status": state.get("safety_status") or SafetyStatus.ALLOWED.value}

    # --- helpers ------------------------------------------------------------------------

    @staticmethod
    def _tool_error_answer(error: dict[str, str]) -> Update:
        code = error.get("code", "")
        if code in messages.TOOL_ERRORS_WITH_DETAILS and error.get("message"):
            text = f"I couldn't do that: {error['message']}"
        else:
            text = messages.TOOL_ERRORS.get(code, messages.DEFAULT_TOOL_ERROR)
        update: Update = {"response": text, "response_source": "template"}
        if code == "access_denied":
            update |= {"safety_status": SafetyStatus.REFUSED.value, "safety_flags": ["authorization_denied"]}
        return update

    async def _llm_answer(
        self,
        prompt: list[ChatMessage],
        *,
        fallback: str,
        sources: list[str],
        required: list[Decimal] | None = None,
    ) -> Update:
        """Ask the LLM to phrase the answer; fall back to the template if anything is off."""
        if self.deps.llm is None:  # pragma: no cover - callers check _llm_answers first
            return {"response": fallback, "response_source": "template"}
        try:
            completion = await self.deps.llm.complete(prompt)
        except LLMError as exc:
            return {"response": fallback, "response_source": "template", "llm_calls": 1, "errors": [exc.code]}

        def rejected(error: str) -> Update:
            return {"response": fallback, "response_source": "template", "llm_calls": 1, "errors": [error]}

        answer = completion.text.strip()
        if not answer:
            return rejected("llm_empty_answer")
        if unsupported := unsupported_numbers(answer, *sources):
            logger.warning("agent.ungrounded_answer", extra={"unsupported_numbers": unsupported[:5]})
            return rejected("ungrounded_answer")
        if unsupported := unsupported_terms(answer, *sources):
            logger.warning("agent.ungrounded_terms", extra={"unsupported_terms": unsupported[:5]})
            return rejected("ungrounded_terms")
        if missing := missing_numbers(answer, required or []):
            logger.warning("agent.incomplete_answer", extra={"missing_numbers": [str(n) for n in missing]})
            return rejected("incomplete_answer")
        return {
            "response": answer,
            "response_source": "llm",
            "llm_calls": 1,
            "llm_latency_ms": completion.latency_ms,
        }
