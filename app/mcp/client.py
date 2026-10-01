"""The agent's gateway to the MCP tool server.

The LangGraph agent (phase 3) never builds MCP requests itself; it calls
``MCPToolGateway.call(tool, arguments, principal=...)``. The gateway:

1. checks the tool exists and validates the arguments with the shared input
   model - LLM-generated arguments are never sent unchecked;
2. signs a short-lived principal token for the *authenticated* customer and
   puts it in the request ``_meta`` (the LLM cannot influence it);
3. calls the server with a timeout;
4. turns every failure into a ``ToolCallError`` with a stable ``code``;
5. validates the response with the shared output model.

The gateway also works with an in-process ``MCPServer`` instead of a URL,
which is how the tests exercise the real protocol without a network.
"""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, RequestParamsMeta, TextContent
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import REQUEST_ID_META_KEY, get_logger, request_id_var
from app.core.security import PRINCIPAL_META_KEY, create_principal_token
from app.mcp.schemas import TOOLS, ToolInput, ToolSpec, describe_validation_error
from app.services.authorization import Principal

logger = get_logger(__name__)

# Failures worth retrying later; everything else is a definitive answer.
RETRYABLE_CODES = frozenset({"unavailable", "timeout"})


def _request_meta(token: str) -> RequestParamsMeta:
    """MCP ``_meta`` for a call: who the call is for, and which request it belongs to."""
    meta = {PRINCIPAL_META_KEY: token}
    if (request_id := request_id_var.get()) is not None:
        meta[REQUEST_ID_META_KEY] = request_id
    # RequestParamsMeta is an *open* TypedDict (PEP 728 extra_items) that accepts
    # custom keys at runtime; mypy does not support that feature yet, hence the cast.
    return cast(RequestParamsMeta, meta)


class ToolCallError(Exception):
    """A tool call that did not produce a valid result."""

    def __init__(self, tool: str, code: str, message: str) -> None:
        super().__init__(f"{tool}: {code}: {message}")
        self.tool = tool
        self.code = code
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_CODES


@dataclass(frozen=True)
class ToolResult:
    tool: str
    arguments: dict[str, Any]  # the validated arguments actually sent (used by evaluation)
    output: BaseModel
    duration_ms: float


class MCPToolGateway:
    def __init__(
        self,
        server: str | MCPServer,
        *,
        auth_secret: str,
        token_ttl_seconds: int = 60,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._server = server
        self._auth_secret = auth_secret
        self._token_ttl_seconds = token_ttl_seconds
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> "MCPToolGateway":
        return cls(
            settings.mcp_server_url,
            auth_secret=settings.require_mcp_auth_secret(),
            token_ttl_seconds=settings.mcp_token_ttl_seconds,
            timeout_seconds=settings.tool_timeout_seconds,
        )

    @staticmethod
    def spec(tool: str) -> ToolSpec:
        try:
            return TOOLS[tool]
        except KeyError:
            raise ToolCallError(tool, "unknown_tool", f"There is no tool named '{tool}'.") from None

    def validate_arguments(self, tool: str, arguments: Mapping[str, Any]) -> ToolInput:
        """Validate (e.g. LLM-generated) arguments without calling anything."""
        try:
            return self.spec(tool).input_model.model_validate(dict(arguments))
        except ValidationError as exc:
            raise ToolCallError(tool, "invalid_arguments", describe_validation_error(exc)) from exc

    async def call(self, tool: str, arguments: Mapping[str, Any], *, principal: Principal) -> ToolResult:
        spec = self.spec(tool)
        payload = self.validate_arguments(tool, arguments).model_dump(mode="json", exclude_none=True)
        token = create_principal_token(
            principal.user_id, secret=self._auth_secret, ttl_seconds=self._token_ttl_seconds
        )

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with Client(self._server, read_timeout_seconds=self._timeout_seconds) as client:
                    result = await client.call_tool(tool, payload, meta=_request_meta(token))
        except TimeoutError as exc:
            logger.warning("mcp.call.timeout", extra={"tool": tool, "timeout_s": self._timeout_seconds})
            raise ToolCallError(
                tool, "timeout", f"The tool did not answer within {self._timeout_seconds:g} seconds."
            ) from exc
        except Exception as exc:
            # Connection refused, DNS failure, protocol error... The SDK wraps these
            # in ExceptionGroups; for the agent they all mean "tools unavailable".
            logger.warning("mcp.call.unavailable", extra={"tool": tool, "error": type(exc).__name__})
            raise ToolCallError(tool, "unavailable", "The banking tools service is unavailable.") from exc
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        if result.is_error:
            error = self._error_from(tool, result)
            logger.info(
                "mcp.call.rejected", extra={"tool": tool, "code": error.code, "duration_ms": duration_ms}
            )
            raise error
        try:
            output = spec.output_model.model_validate(result.structured_content)
        except ValidationError as exc:
            raise ToolCallError(tool, "invalid_tool_output", describe_validation_error(exc)) from exc

        logger.info("mcp.call.completed", extra={"tool": tool, "duration_ms": duration_ms})
        return ToolResult(tool=tool, arguments=payload, output=output, duration_ms=duration_ms)

    @staticmethod
    def _error_from(tool: str, result: CallToolResult) -> ToolCallError:
        structured = result.structured_content or {}
        error = structured.get("error") if isinstance(structured, dict) else None
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return ToolCallError(tool, error["code"], str(error.get("message", "")))
        # Errors produced by the SDK itself (not by our tools) arrive as plain text.
        text = " ".join(block.text for block in result.content if isinstance(block, TextContent))
        return ToolCallError(tool, "tool_error", text[:300] or "The tool reported an error.")
