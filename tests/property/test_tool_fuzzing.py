"""Fuzzing the MCP tool boundary (Hypothesis).

An attacker who reaches the tool server does not have to send sensible arguments.
For every tool, Hypothesis sends Nino's valid token with two kinds of calls:

* **well-formed** calls generated from the tool's own JSON schema (``hypothesis-jsonschema``),
  often aimed at one of the victim's accounts. Random junk alone almost never gets past
  validation, so it never reaches the ownership checks: with junk only, this test did not
  notice a deleted ownership check (mutation testing showed it);
* **junk**: any declared field with any value, smuggled extra fields (``user_id``), SQL
  fragments, huge numbers, nested lists and objects.

Whatever arrives, two things must hold:
* no data of another customer comes back (beyond what the caller sent in themselves);
* the server answers with a result or a *named* error - never ``internal_error``,
  which would mean an input reached code that did not expect it (a bug).

The calls bypass the agent-side gateway on purpose, as a real attacker would.
"""

import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema
from mcp.server.mcpserver import MCPServer

from app.core.security import create_principal_token
from app.mcp.schemas import TOOLS
from tests.constants import GIORGI_CHECKING, NINO, NINO_CHECKING, TEST_MCP_SECRET
from tests.safety.test_tool_authorization import VICTIM_ACCOUNTS, VICTIM_MARKERS, raw_call

INTERESTING = [
    str(GIORGI_CHECKING),
    str(NINO_CHECKING),
    VICTIM_ACCOUNTS[0].account_number,
    "CUST-0002",
    "' OR 1=1 --",
    "%",
    "_",
    "\x00",
    "custom",
    "last_month",
    "2026-02-30",
    "0001-01-01",
    "9999-12-31",
    "groceries",
    "GEL",
    "USD",
    "debit",
    "consumer_loan",
    "1e309",
]
SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**15), 10**15),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=30),
    st.sampled_from(INTERESTING),
)
VALUES = st.recursive(
    SCALARS,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=6), inner, max_size=3),
)


WELL_FORMED = {name: from_schema(spec.input_model.model_json_schema()) for name, spec in TOOLS.items()}
TARGET_ACCOUNTS = [str(a.id) for a in VICTIM_ACCOUNTS] + [str(NINO_CHECKING)]


@st.composite
def tool_calls(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    name = draw(st.sampled_from(sorted(TOOLS)))
    fields = sorted(TOOLS[name].input_model.model_fields)
    if draw(st.booleans()):
        arguments: dict[str, Any] = draw(WELL_FORMED[name])
        if "account_id" in fields and draw(st.booleans()):
            arguments["account_id"] = draw(st.sampled_from(TARGET_ACCOUNTS))
        return name, arguments
    keys = st.sampled_from([*fields, "user_id", "customer_id", "principal"])
    return name, draw(st.dictionaries(keys, VALUES, max_size=5))


TOKEN = create_principal_token(NINO, secret=TEST_MCP_SECRET)


# The tools only read, so the examples can share one seeded database (see
# tests/safety/test_tool_authorization.py::test_no_tool_changes_the_database).
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(tool_calls())
async def test_arbitrary_arguments_never_leak_or_crash(tool_server: MCPServer, call: tuple[str, Any]) -> None:
    name, arguments = call
    result = await raw_call(tool_server, name, arguments, TOKEN)

    text = " ".join(getattr(block, "text", "") for block in result.content)
    sent = json.dumps(arguments, default=str)
    leaked = [m for m in VICTIM_MARKERS if m in text and m not in sent]
    assert not leaked, f"{name}({sent}) returned another customer's data: {leaked}"
    if result.is_error and result.structured_content:
        assert result.structured_content["error"]["code"] != "internal_error", f"{name}({sent}): {text}"
