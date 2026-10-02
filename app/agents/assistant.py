"""``BankingAssistant``: the one entry point to the agent.

The API (phase 8), the CLI and the evaluation framework (phase 6) all call
``respond()`` and get an ``AssistantReply``: the answer plus everything needed
to judge it - intent, tool, arguments, whether the LLM or a template wrote the
answer, safety status, and per-node timings.

Conversations with a pending confirmation
-----------------------------------------
A high-risk request ("Transfer 5000 GEL...") pauses the graph and the reply
carries ``pending_confirmation``. The caller passes it back with the customer's
next message (just like ``previous``):

* "yes" / "no"         -> the paused run is resumed and finishes;
* anything else        -> the pending request is cancelled and the new message is
                          answered normally, with a short notice;
* another customer, an unknown or a stale thread -> ignored; the message is a new turn.

Every turn runs on its own LangGraph thread (a fresh ID, never the caller's
request ID), so no state leaks from one turn into the next. Finished threads are
deleted; paused ones are kept until they are answered or go stale.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
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
from app.safety import actions
from app.safety.actions import ActionRequest, ProposedAction
from app.safety.output_guard import OutputGuard

logger = get_logger(__name__)

# Paused threads are kept this long after their confirmation expired (so a late "yes" gets
# "that expired" rather than being treated as a new question), then deleted.
STALE_AFTER = timedelta(hours=1)


@dataclass(frozen=True)
class PreviousTurn:
    """What the agent remembers from the last turn, so follow-ups like 'and last month?' work."""

    intent: Intent
    arguments: dict[str, Any] = field(default_factory=dict)


class PendingConfirmation(BaseModel):
    """A high-risk request waiting for "yes" or "no". Pass it back with the next message."""

    thread_id: str
    action: ProposedAction
    expires_at: datetime


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
    pending_confirmation: PendingConfirmation | None = None
    confirmation: str | None = None  # confirmed | declined | expired | superseded | rejected
    action_request: ActionRequest | None = None  # the MOCK record after a confirmed action

    def as_previous_turn(self) -> PreviousTurn | None:
        if self.intent in INTENT_TOOL and self.tool_error is None:
            return PreviousTurn(self.intent, self.tool_arguments or {})
        return None


@dataclass(frozen=True)
class _Paused:
    user_id: uuid.UUID
    expires_at: datetime


class BankingAssistant:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self._checkpointer = InMemorySaver()
        self.graph = build_agent_graph(AgentNodes(deps), checkpointer=self._checkpointer)
        self._paused: dict[str, _Paused] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        gateway: MCPToolGateway,
        llm: LLMClient | None = None,
        extra_secrets: Sequence[str] = (),
    ) -> "BankingAssistant":
        config = AgentConfig(
            router_mode=settings.router_mode,
            response_mode=settings.response_mode,
            max_message_chars=settings.max_message_chars,
        )
        # Configured secret values must never appear in an answer, whatever produced it.
        secrets = [
            secret.get_secret_value()
            for secret in (settings.mcp_auth_secret, settings.llm_api_key, settings.embedding_api_key)
            if secret is not None
        ] + list(extra_secrets)
        return cls(
            AgentDeps(
                gateway=gateway,
                llm=llm or create_llm_client(settings),
                config=config,
                output_guard=OutputGuard(secrets),
            )
        )

    async def aclose(self) -> None:
        """Close the LLM client's connections (call on application shutdown)."""
        if self.deps.llm is not None:
            await self.deps.llm.aclose()

    @property
    def paused_threads(self) -> int:
        return len(self._paused)

    async def respond(
        self,
        *,
        user_id: uuid.UUID,
        message: str,
        request_id: str | None = None,
        channel: Channel = Channel.TEXT,
        previous: PreviousTurn | None = None,
        pending: PendingConfirmation | None = None,
    ) -> AssistantReply:
        request_id = request_id or uuid.uuid4().hex
        trace = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            reply = await self._respond(user_id, message, request_id, channel, previous, pending, started)
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
                "safety_flags": reply.safety_flags,
                "llm_calls": reply.llm_calls,
                "latency_ms": reply.latency_ms,
            },
        )
        return reply

    async def _respond(
        self,
        user_id: uuid.UUID,
        message: str,
        request_id: str,
        channel: Channel,
        previous: PreviousTurn | None,
        pending: PendingConfirmation | None,
        started: float,
    ) -> AssistantReply:
        await self._drop_stale_threads()
        notice = None
        if pending is not None and (resumed := await self._resume(pending, user_id, message)) is not None:
            if resumed.get("confirmation") != "superseded":
                return self._reply(resumed, request_id, started, thread_id=None)
            notice = actions.SUPERSEDED_NOTICE

        thread_id = uuid.uuid4().hex
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
        final: dict[str, Any] = await self.graph.ainvoke(initial, self._thread(thread_id))
        reply = self._reply(final, request_id, started, thread_id=thread_id)
        if reply.pending_confirmation is None:
            await self._checkpointer.adelete_thread(thread_id)
        else:
            self._paused[thread_id] = _Paused(user_id, reply.pending_confirmation.expires_at)
        if notice:
            reply = reply.model_copy(update={"response": f"{notice}\n{reply.response}"})
        return reply

    async def _resume(
        self, pending: PendingConfirmation, user_id: uuid.UUID, message: str
    ) -> dict[str, Any] | None:
        """Finish a paused run with the customer's answer. ``None`` if there is nothing to resume."""
        paused = self._paused.get(pending.thread_id)
        if paused is None or paused.user_id != user_id:
            # Unknown, already answered, or someone else's: never touch another customer's thread.
            logger.warning("safety.unknown_pending_confirmation", extra={"exists": paused is not None})
            return None
        del self._paused[pending.thread_id]
        resume: Command[str] = Command(resume={"user_id": str(user_id), "message": message})
        final: dict[str, Any] = await self.graph.ainvoke(resume, self._thread(pending.thread_id))
        await self._checkpointer.adelete_thread(pending.thread_id)
        return final

    async def _drop_stale_threads(self) -> None:
        now = self.deps.clock()
        for thread_id, paused in list(self._paused.items()):
            if now > paused.expires_at + STALE_AFTER:
                del self._paused[thread_id]
                await self._checkpointer.adelete_thread(thread_id)

    @staticmethod
    def _thread(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _reply(
        final: dict[str, Any], request_id: str, started: float, *, thread_id: str | None
    ) -> AssistantReply:
        tool_error = final.get("tool_error")
        pending = None
        if thread_id and (interrupts := final.get("__interrupt__")):
            action = ProposedAction.model_validate(interrupts[0].value["action"])
            if action.expires_at is None:  # pragma: no cover - high_risk_action always sets it
                raise RuntimeError("paused without an expiry time")
            pending = PendingConfirmation(thread_id=thread_id, action=action, expires_at=action.expires_at)
        action_request = final.get("action_request")
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
            pending_confirmation=pending,
            confirmation=final.get("confirmation"),
            action_request=ActionRequest.model_validate(action_request) if action_request else None,
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
