"""Write the agent's LangGraph as a Mermaid diagram to docs/agent_graph.md.

    python -m scripts.export_agent_graph

The diagram is generated from the compiled graph itself, so it cannot drift
from the code; a test fails if the committed file is out of date.
"""

from pathlib import Path

from app.agents.graph import build_agent_graph
from app.agents.nodes import AgentDeps, AgentNodes
from app.mcp.client import MCPToolGateway

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "agent_graph.md"


def render_document() -> str:
    # Nothing is called while drawing; the gateway only has to exist.
    nodes = AgentNodes(AgentDeps(gateway=MCPToolGateway("http://unused", auth_secret="-" * 32)))
    mermaid = build_agent_graph(nodes).get_graph().draw_mermaid()
    return (
        "# BankAI agent graph\n\n"
        "Generated from the compiled LangGraph by `python -m scripts.export_agent_graph`"
        " - do not edit by hand.\n"
        "Dotted arrows are conditional edges (the routing decisions).\n\n"
        f"```mermaid\n{mermaid.strip()}\n```\n"
    )


def main() -> int:
    OUTPUT.write_text(render_document(), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
