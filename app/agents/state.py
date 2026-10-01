"""The typed state that flows through the LangGraph agent.

LangGraph passes one state dict from node to node. Each node returns only the
keys it changes, and LangGraph merges them in. Keys with a *reducer*
(``Annotated[..., reducer]``) are combined instead of overwritten - so every
node can append to ``errors`` or add its own entry to ``timings_ms``.

Values are kept JSON-friendly (strings, numbers, dicts), so the state can be
checkpointed later (phase 5: pausing the graph for a human confirmation).
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
    intent_source: str  # rules | llm | follow_up | fallback
    intent_confidence: float
    route_reason: str

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
    llm_latency_ms: float | None

    # --- safety & diagnostics (accumulated across nodes) ---
    safety_status: str
    safety_flags: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    timings_ms: Annotated[dict[str, float], merge_dicts]
    llm_calls: Annotated[int, operator.add]
