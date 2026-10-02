"""The typed state that flows through the LangGraph agent.

LangGraph passes one state dict from node to node. Each node returns only the
keys it changes, and LangGraph merges them in. Keys with a *reducer*
(``Annotated[..., reducer]``) are combined instead of overwritten - so every
node can append to ``errors`` or add its own entry to ``timings_ms``.

Values are kept JSON-friendly (strings, numbers, dicts), so the state can be
checkpointed: the graph pauses there while it waits for a human confirmation.
"""

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {**left, **right}


class AgentState(TypedDict, total=False):
    # --- input (set by the caller, never by the LLM) ---
    request_id: str
    user_id: str  # the authenticated customer
    message: str
    channel: str  # "text" | "voice"
    previous_intent: str | None  # the previous turn, for follow-ups ("and last month?")
    previous_arguments: dict[str, Any]

    # --- input validation ---
    language: str  # "en" | "ka"
    input_error: str | None

    # --- routing ---
    intent: str
    intent_source: str  # safety | rules | llm | follow_up | fallback
    intent_confidence: float
    route_reason: str  # safety category, action kind, or the router's reason
    safety_rule: str | None  # the input-guard rule that fired

    # --- tool call ---
    selected_tool: str | None
    tool_arguments: dict[str, Any] | None
    missing_arguments: list[str]
    tool_result: dict[str, Any] | None
    tool_error: dict[str, str] | None
    tool_latency_ms: float | None

    # --- response ---
    facts: list[str]
    template_answer: str
    key_numbers: list[str]  # Decimals as strings (JSON-friendly)
    llm_may_rephrase: bool
    prompt_kind: str  # facts | passages
    sources: list[str]
    response: str
    response_source: str  # llm | template
    fallback_response: str | None  # the template answer, in case the output guard rejects the LLM's
    llm_latency_ms: float | None

    # --- high-risk actions (human in the loop) ---
    pending_action: dict[str, Any] | None  # ProposedAction, waiting for "yes" / "no"
    confirmation: str | None  # confirmed | declined | expired | superseded | rejected
    action_request: dict[str, Any] | None  # the MOCK request recorded after "yes"

    # --- safety & diagnostics (accumulated across nodes) ---
    safety_status: str
    safety_flags: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    timings_ms: Annotated[dict[str, float], merge_dicts]
    llm_calls: Annotated[int, operator.add]
