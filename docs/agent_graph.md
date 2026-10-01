# BankAI agent graph

Generated from the compiled LangGraph by `python -m scripts.export_agent_graph` - do not edit by hand.
Dotted arrows are conditional edges (the routing decisions).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	validate_input(validate_input)
	classify_intent(classify_intent)
	call_tool(call_tool)
	validate_tool_result(validate_tool_result)
	generate_response(generate_response)
	clarify(clarify)
	general_response(general_response)
	refuse(refuse)
	high_risk_action(high_risk_action)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> validate_input;
	call_tool --> validate_tool_result;
	clarify --> finalize;
	classify_intent -.-> call_tool;
	classify_intent -.-> clarify;
	classify_intent -.-> general_response;
	classify_intent -.-> high_risk_action;
	classify_intent -.-> refuse;
	general_response --> finalize;
	generate_response --> finalize;
	high_risk_action --> finalize;
	refuse --> finalize;
	validate_input -. &nbsp;valid&nbsp; .-> classify_intent;
	validate_input -. &nbsp;invalid&nbsp; .-> finalize;
	validate_tool_result --> generate_response;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
