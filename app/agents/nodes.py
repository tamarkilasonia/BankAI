"""The LangGraph nodes - each one a small, testable step of the agent.

Every node receives the whole ``AgentState`` and returns only the keys it
changes. Nodes never raise for expected problems (bad input, tool errors, LLM
failures): they record them in the state and the graph routes to a safe answer.
"""

import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from app.agents import messages
from app.agents.grounding import missing_numbers, unsupported_numbers
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
from app.services.authorization import Principal

logger = get_logger(__name__)

_GEORGIAN = re.compile(f"[{chr(0x10A0)}-{chr(0x10FF)}]")  # Georgian script block
_PERIOD_KEYS = ("period", "start_date", "end_date")

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

    # --- 2. routing ---------------------------------------------------------------------

    async def classify_intent(self, state: AgentState) -> Update:
        previous = None
        if previous_intent := state.get("previous_intent"):
            previous = RouteDecision(Intent(previous_intent), dict(state.get("previous_arguments") or {}))
        outcome = await self.router.route(state["message"], previous=previous)
        decision = outcome.decision

        update: Update = {
            "intent": decision.intent.value,
            "intent_source": decision.source,
            "intent_confidence": decision.confidence,
            "route_reason": decision.reason,
            "llm_calls": outcome.llm_calls,
            "errors": list(outcome.errors),
            "selected_tool": None,
            "tool_arguments": None,
            "missing_arguments": [],
        }
        if outcome.llm_latency_ms is not None:
            update["llm_latency_ms"] = outcome.llm_latency_ms
        if tool := INTENT_TOOL.get(decision.intent):
            proposed = decision.arguments
            if tool == "search_banking_policy":
                # Search with the customer's own words, not an LLM paraphrase of them.
                proposed = {"query": state["message"]}
            args, missing, dropped = prepare_arguments(tool, proposed)
            update |= {
                "selected_tool": tool,
                "tool_arguments": args,
                "missing_arguments": list(decision.missing) or missing,
            }
            update["errors"] += [f"dropped_argument:{key}" for key in dropped]
        if decision.intent is Intent.UNSAFE:
            update["safety_flags"] = [decision.reason or "unsafe_request"]
        elif decision.intent is Intent.HIGH_RISK_ACTION:
            update["safety_flags"] = ["high_risk_action"]
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
        rendered = render(output)
        return {
            "facts": rendered.facts,
            "template_answer": rendered.answer,
            "key_numbers": [str(n) for n in rendered.key_numbers],
            "llm_may_rephrase": rendered.llm_may_rephrase,
            "prompt_kind": rendered.prompt_kind,
            "sources": list(rendered.sources),
        }

    # --- 4. answers ---------------------------------------------------------------------

    async def generate_response(self, state: AgentState) -> Update:
        if error := state.get("tool_error"):
            return self._tool_error_answer(error)
        template = state["template_answer"]
        if not self._llm_answers or not state.get("llm_may_rephrase", True):
            update: Update = {"response": template, "response_source": "template"}
        else:
            facts = state.get("facts", [])
            language = state.get("language", "en")
            if state.get("prompt_kind") == "passages":
                prompt = knowledge_messages(state["message"], facts, language)
            else:
                prompt = responder_messages(state["message"], facts, language)
            required = [Decimal(n) for n in state.get("key_numbers", [])]
            update = await self._llm_answer(
                prompt, fallback=template, sources=[*facts, state["message"]], required=required
            )
        return update | {"response": with_sources(update["response"], update["response_source"], state)}

    async def general_response(self, state: AgentState) -> Update:
        if state.get("intent_source") == "fallback":
            return {"response": messages.NOT_UNDERSTOOD, "response_source": "template"}
        if not self._llm_answers:
            return {"response": messages.GREETING, "response_source": "template"}
        prompt = general_messages(state["message"], state.get("language", "en"))
        return await self._llm_answer(prompt, fallback=messages.GREETING, sources=[state["message"]])

    async def clarify(self, state: AgentState) -> Update:
        text = messages.CLARIFICATIONS.get(state.get("selected_tool") or "", messages.DEFAULT_CLARIFICATION)
        return {"response": text, "response_source": "template"}

    async def refuse(self, state: AgentState) -> Update:
        text = messages.REFUSALS.get(state.get("route_reason", ""), messages.DEFAULT_REFUSAL)
        return {"response": text, "response_source": "template", "safety_status": SafetyStatus.REFUSED.value}

    async def high_risk_action(self, state: AgentState) -> Update:
        # Phase 5 turns this into an explicit confirmation flow (still without moving money).
        return {
            "response": messages.HIGH_RISK,
            "response_source": "template",
            "safety_status": SafetyStatus.REFUSED.value,
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

        answer = completion.text.strip()
        if not answer:
            return {
                "response": fallback,
                "response_source": "template",
                "llm_calls": 1,
                "errors": ["llm_empty_answer"],
            }
        if unsupported := unsupported_numbers(answer, *sources):
            logger.warning("agent.ungrounded_answer", extra={"unsupported_numbers": unsupported[:5]})
            return {
                "response": fallback,
                "response_source": "template",
                "llm_calls": 1,
                "errors": ["ungrounded_answer"],
            }
        if missing := missing_numbers(answer, required or []):
            logger.warning("agent.incomplete_answer", extra={"missing_numbers": [str(n) for n in missing]})
            return {
                "response": fallback,
                "response_source": "template",
                "llm_calls": 1,
                "errors": ["incomplete_answer"],
            }
        return {
            "response": answer,
            "response_source": "llm",
            "llm_calls": 1,
            "llm_latency_ms": completion.latency_ms,
        }
