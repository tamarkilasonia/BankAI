"""``BankingAssistant``: the one entry point to the agent.

The API (phase 8), the CLI and the evaluation framework (phase 6) all call
``respond()`` and get an ``AssistantReply``: the answer plus everything needed
to judge it - intent, tool, arguments, whether the LLM or a template wrote the
answer, safety status, and per-node timings.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agents import messages
from app.agents.graph import build_agent_graph
from app.agents.intents import INTENT_TOOL, Intent
from app.agents.nodes import AgentConfig, AgentDeps, AgentNodes
from app.agents.state import AgentState
from app.core.config import Settings
from app.core.logging import get_logger, request_id_var
from app.llm import LLMClient, create_llm_client
from app.mcp.client import MCPToolGateway
from app.models.enums import Channel, SafetyStatus

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreviousTurn:
    """What the agent remembers from the last turn, so follow-ups like 'and last month?' work."""

    intent: Intent
    arguments: dict[str, Any] = field(default_factory=dict)


class AssistantReply(BaseModel):
    request_id: str
    response: str
    intent: Intent
    intent_source: str
    intent_confidence: float
    tool: str | None
    tool_arguments: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_error: str | None  # error code, never internal details
    response_source: str  # "llm" | "template"
    safety_status: SafetyStatus | None  # None only if the request failed unexpectedly
    safety_flags: list[str]
    language: str
    llm_calls: int
    errors: list[str]
    latency_ms: float
    timings_ms: dict[str, float]

    def as_previous_turn(self) -> PreviousTurn | None:
        if self.intent in INTENT_TOOL and self.tool_error is None:
            return PreviousTurn(self.intent, self.tool_arguments or {})
        return None


class BankingAssistant:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.graph = build_agent_graph(AgentNodes(deps))

    @classmethod
    def from_settings(
        cls, settings: Settings, *, gateway: MCPToolGateway, llm: LLMClient | None = None
    ) -> "BankingAssistant":
        config = AgentConfig(
            router_mode=settings.router_mode,
            response_mode=settings.response_mode,
            max_message_chars=settings.max_message_chars,
        )
        return cls(AgentDeps(gateway=gateway, llm=llm or create_llm_client(settings), config=config))

    async def aclose(self) -> None:
        """Close the LLM client's connections (call on application shutdown)."""
        if self.deps.llm is not None:
            await self.deps.llm.aclose()

    async def respond(
        self,
        *,
        user_id: uuid.UUID,
        message: str,
        request_id: str | None = None,
        channel: Channel = Channel.TEXT,
        previous: PreviousTurn | None = None,
    ) -> AssistantReply:
        request_id = request_id or uuid.uuid4().hex
        trace = request_id_var.set(request_id)
        started = time.perf_counter()
        initial: AgentState = {
            "request_id": request_id,
            "user_id": str(user_id),  # from authentication - never from the LLM
            "message": message,
            "channel": channel.value,
            "previous_intent": previous.intent.value if previous else None,
            "previous_arguments": previous.arguments if previous else {},
            "safety_flags": [],
            "errors": [],
            "timings_ms": {},
            "llm_calls": 0,
        }
        try:
            final: dict[str, Any] = await self.graph.ainvoke(initial)
            reply = self._reply(final, request_id, started)
        except Exception:
            # A bug, not an expected failure (those are handled inside the graph).
            logger.exception("agent.unexpected_error")
            reply = self._internal_error(request_id, started)
        finally:
            request_id_var.reset(trace)

        logger.info(
            "agent.completed",
            extra={
                "request_id": request_id,
                "user_id": str(user_id),
                "intent": reply.intent,
                "intent_source": reply.intent_source,
                "tool": reply.tool,
                "tool_error": reply.tool_error,
                "response_source": reply.response_source,
                "safety_status": reply.safety_status,
                "llm_calls": reply.llm_calls,
                "latency_ms": reply.latency_ms,
            },
        )
        return reply

    @staticmethod
    def _reply(final: dict[str, Any], request_id: str, started: float) -> AssistantReply:
        tool_error = final.get("tool_error")
        return AssistantReply(
            request_id=request_id,
            response=final.get("response") or messages.INTERNAL_ERROR,
            intent=Intent(final.get("intent") or Intent.GENERAL),
            intent_source=final.get("intent_source")
            or ("validation" if final.get("input_error") else "none"),
            intent_confidence=final.get("intent_confidence", 0.0),
            tool=final.get("selected_tool"),
            tool_arguments=final.get("tool_arguments"),
            tool_result=final.get("tool_result"),
            tool_error=tool_error.get("code") if tool_error else None,
            response_source=final.get("response_source") or "template",
            safety_status=SafetyStatus(final.get("safety_status") or SafetyStatus.ALLOWED),
            safety_flags=final.get("safety_flags", []),
            language=final.get("language", "en"),
            llm_calls=final.get("llm_calls", 0),
            errors=final.get("errors", [])
            + ([f"input:{final['input_error']}"] if final.get("input_error") else []),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            timings_ms=final.get("timings_ms", {}),
        )

    @staticmethod
    def _internal_error(request_id: str, started: float) -> AssistantReply:
        return AssistantReply(
            request_id=request_id,
            response=messages.INTERNAL_ERROR,
            intent=Intent.GENERAL,
            intent_source="error",
            intent_confidence=0.0,
            tool=None,
            tool_arguments=None,
            tool_result=None,
            tool_error=None,
            response_source="template",
            safety_status=None,
            safety_flags=[],
            language="en",
            llm_calls=0,
            errors=["internal_error"],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            timings_ms={},
        )
