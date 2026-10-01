"""The LangGraph workflow: nodes + the edges that connect them.

    START -> validate_input --(invalid)----------------------------------> finalize -> END
                 | (valid)
             classify_intent
                 |--(tool intent)--> call_tool -> validate_tool_result -> generate_response -> finalize
                 |    (incl. policy questions: search_banking_policy = RAG over the knowledge base)
                 |--(missing info)--> clarify ---------------------------------------------> finalize
                 |--(general)-------> general_response ------------------------------------> finalize
                 |--(unsafe)--------> refuse ----------------------------------------------> finalize
                 `--(high risk)-----> high_risk_action ------------------------------------> finalize

Phase 5 adds an input guard before classification, an output guard before
``finalize`` and a human-in-the-loop confirmation for high-risk actions.
"""

import time
from collections.abc import Awaitable
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import AgentNodes, Update
from app.agents.state import AgentState


class Node(Protocol):
    """A graph step. LangGraph calls it with the parameter named ``state``."""

    def __call__(self, state: AgentState) -> Awaitable[Update]: ...


# All the nodes that write the final answer.
ANSWER_NODES = ("generate_response", "clarify", "general_response", "refuse", "high_risk_action")


def timed(name: str, node: Node) -> Node:
    """Record each node's duration in ``state['timings_ms']`` (observability)."""

    async def run(state: AgentState) -> Update:
        started = time.perf_counter()
        update = await node(state)
        return {**update, "timings_ms": {name: round((time.perf_counter() - started) * 1000, 2)}}

    return run


def build_agent_graph(nodes: AgentNodes) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph = StateGraph(AgentState)
    steps: dict[str, Node] = {
        "validate_input": nodes.validate_input,
        "classify_intent": nodes.classify_intent,
        "call_tool": nodes.call_tool,
        "validate_tool_result": nodes.validate_tool_result,
        "generate_response": nodes.generate_response,
        "clarify": nodes.clarify,
        "general_response": nodes.general_response,
        "refuse": nodes.refuse,
        "high_risk_action": nodes.high_risk_action,
        "finalize": nodes.finalize,
    }
    for name, node in steps.items():
        graph.add_node(name, timed(name, node))

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input", nodes.after_validation, {"valid": "classify_intent", "invalid": "finalize"}
    )
    graph.add_conditional_edges(
        "classify_intent",
        nodes.after_classification,
        {name: name for name in ("call_tool", *ANSWER_NODES) if name != "generate_response"},
    )
    graph.add_edge("call_tool", "validate_tool_result")
    graph.add_edge("validate_tool_result", "generate_response")
    for name in ANSWER_NODES:
        graph.add_edge(name, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
