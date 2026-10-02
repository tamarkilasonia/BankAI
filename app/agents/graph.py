"""The LangGraph workflow: nodes + the edges that connect them.

    START -> validate_input --(invalid)----------------------------------------------> output_guard
                 | (valid)
             input_guard --(refuse)--> refuse ---------------------------------------> output_guard
                 |       `-(confirm)-> high_risk_action --(credentials: refuse)------> output_guard
                 | (allow)                    `--> confirm_action [pause: "yes"/"no"] -> output_guard
             classify_intent
                 |--(tool intent)--> call_tool -> validate_tool_result -> generate_response -> output_guard
                 |    (incl. policy questions: search_banking_policy = RAG over the knowledge base)
                 |--(missing info)--> clarify -----------------------------------------------> output_guard
                 |--(general)-------> general_response --------------------------------------> output_guard
                 |--(unsafe)--------> refuse ------------------------------------------------> output_guard
                 `--(high risk)-----> high_risk_action -> (as above)

    output_guard -> finalize -> END

``confirm_action`` pauses the run with LangGraph's ``interrupt``; that needs a
*checkpointer* (where the paused state is saved). ``BankingAssistant`` passes an
in-memory one and resumes the run when the customer answers.
"""

import time
from collections.abc import Awaitable
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import AgentNodes, Update
from app.agents.state import AgentState


class Node(Protocol):
    """A graph step. LangGraph calls it with the parameter named ``state``."""

    def __call__(self, state: AgentState) -> Awaitable[Update]: ...


# All the nodes that write the final answer (each one then goes through the output guard).
ANSWER_NODES = ("generate_response", "clarify", "general_response", "refuse", "confirm_action")


def timed(name: str, node: Node) -> Node:
    """Record each node's duration in ``state['timings_ms']`` (observability)."""

    async def run(state: AgentState) -> Update:
        started = time.perf_counter()
        update = await node(state)
        return {**update, "timings_ms": {name: round((time.perf_counter() - started) * 1000, 2)}}

    return run


def build_agent_graph(
    nodes: AgentNodes, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph = StateGraph(AgentState)
    steps: dict[str, Node] = {
        "validate_input": nodes.validate_input,
        "input_guard": nodes.input_guard,
        "classify_intent": nodes.classify_intent,
        "call_tool": nodes.call_tool,
        "validate_tool_result": nodes.validate_tool_result,
        "generate_response": nodes.generate_response,
        "clarify": nodes.clarify,
        "general_response": nodes.general_response,
        "refuse": nodes.refuse,
        "high_risk_action": nodes.high_risk_action,
        "confirm_action": nodes.confirm_action,
        "output_guard": nodes.output_guard,
        "finalize": nodes.finalize,
    }
    for name, node in steps.items():
        graph.add_node(name, timed(name, node))

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input", nodes.after_validation, {"valid": "input_guard", "invalid": "output_guard"}
    )
    graph.add_conditional_edges(
        "input_guard",
        nodes.after_input_guard,
        {"allow": "classify_intent", "refuse": "refuse", "confirm": "high_risk_action"},
    )
    graph.add_conditional_edges(
        "classify_intent",
        nodes.after_classification,
        {name: name for name in ("call_tool", "clarify", "general_response", "refuse", "high_risk_action")},
    )
    graph.add_edge("call_tool", "validate_tool_result")
    graph.add_edge("validate_tool_result", "generate_response")
    graph.add_conditional_edges(
        "high_risk_action",
        nodes.after_high_risk_action,
        {"confirm": "confirm_action", "done": "output_guard"},
    )
    for name in ANSWER_NODES:
        graph.add_edge(name, "output_guard")
    graph.add_edge("output_guard", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
