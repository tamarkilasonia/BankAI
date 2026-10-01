"""BankAI MCP server: the banking services exposed as MCP tools.

Why a separate MCP server?
    The agent never touches the database. It only knows a protocol: "list
    tools, call tool X with JSON arguments". All data access, input validation
    and authorization live behind that boundary, in a process that can be
    deployed, secured and scaled independently - and any MCP-compatible client
    (another agent, an IDE, a test harness) can reuse the same tools.

Every tool call goes through ``_run_tool``:

    1. authenticate  - verify the signed principal token from the request ``_meta``
    2. validate      - parse arguments with the tool's Pydantic input model
    3. execute       - call the domain service in a fresh DB session
    4. respond       - structured JSON result, or a structured error
                       ``{"error": {"code", "message"}}`` that never leaks internals

All tools are read-only: there is no tool that can change a balance.

Run it with ``python -m app.mcp`` (streamable HTTP on MCP_HOST:MCP_PORT).
"""

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.logging import (
    REQUEST_ID_META_KEY,
    configure_logging,
    get_logger,
    request_id_var,
    safe_request_id,
)
from app.core.security import PRINCIPAL_META_KEY, InvalidTokenError, verify_principal_token
from app.core.time import utcnow
from app.database.session import get_sessionmaker
from app.mcp.schemas import (
    TOOLS,
    AnalyzeSpendingInput,
    CalculateLoanPaymentInput,
    GetAccountBalanceInput,
    GetAccountDetailsInput,
    GetExchangeRateInput,
    GetTransactionsInput,
    PeriodInput,
    SearchBankingPolicyInput,
    SearchTransactionsInput,
    ToolSpec,
    describe_validation_error,
)
from app.rag.embeddings import EmbeddingError
from app.rag.factory import create_retriever
from app.rag.retriever import Retriever
from app.rag.schemas import PolicyPassage, PolicySearchResult
from app.services import accounts, exchange_rates, loans, spending, transactions
from app.services.authorization import Principal
from app.services.errors import DomainError, ServiceUnavailableError
from app.services.periods import DateRange, resolve_period

logger = get_logger(__name__)

SERVER_NAME = "bankai-banking-tools"
SERVER_INSTRUCTIONS = (
    "Read-only tools over MOCK banking data. Every call must carry a principal token in "
    f"_meta['{PRINCIPAL_META_KEY}']; tools only ever return the token holder's own data."
)
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


@dataclass(frozen=True)
class ToolServerDeps:
    """Everything the tools need, injected so tests can swap the database, clock and retriever."""

    session_factory: async_sessionmaker[AsyncSession]
    auth_secret: str
    clock: Callable[[], datetime] = field(default=utcnow)
    retriever: Retriever | None = None  # None: search_banking_policy reports "unavailable"


# A handler adapts validated tool arguments to a service call.
Handler = Callable[[AsyncSession, Principal, Any, datetime], Awaitable[BaseModel]]


def _period(args: PeriodInput, now: datetime) -> DateRange:
    return resolve_period(args.period, now=now, start_date=args.start_date, end_date=args.end_date)


async def _get_account_balance(
    session: AsyncSession, principal: Principal, args: GetAccountBalanceInput, now: datetime
) -> BaseModel:
    return await accounts.get_account_balance(
        session,
        principal,
        now=now,
        account_id=args.account_id,
        account_type=args.account_type,
        currency=args.currency,
    )


async def _get_account_details(
    session: AsyncSession, principal: Principal, args: GetAccountDetailsInput, now: datetime
) -> BaseModel:
    return await accounts.get_account_details(session, principal, account_id=args.account_id)


async def _get_transactions(
    session: AsyncSession, principal: Principal, args: GetTransactionsInput, now: datetime
) -> BaseModel:
    return await transactions.get_transactions(
        session,
        principal,
        date_range=_period(args, now),
        account_id=args.account_id,
        category=args.category,
        direction=args.direction,
        limit=args.limit,
    )


async def _search_transactions(
    session: AsyncSession, principal: Principal, args: SearchTransactionsInput, now: datetime
) -> BaseModel:
    return await transactions.search_transactions(
        session,
        principal,
        date_range=_period(args, now),
        text=args.text,
        min_amount=args.min_amount,
        max_amount=args.max_amount,
        category=args.category,
        account_id=args.account_id,
        limit=args.limit,
    )


async def _analyze_spending(
    session: AsyncSession, principal: Principal, args: AnalyzeSpendingInput, now: datetime
) -> BaseModel:
    return await spending.analyze_spending(
        session,
        principal,
        date_range=_period(args, now),
        currency=args.currency,
        category=args.category,
        account_id=args.account_id,
        top_merchants=args.top_merchants,
    )


async def _get_exchange_rate(
    session: AsyncSession, principal: Principal, args: GetExchangeRateInput, now: datetime
) -> BaseModel:
    return exchange_rates.get_exchange_rate(args.base_currency, args.quote_currency, args.amount)


async def _calculate_loan_payment(
    session: AsyncSession, principal: Principal, args: CalculateLoanPaymentInput, now: datetime
) -> BaseModel:
    return await loans.calculate_loan_payment(
        session, product_code=args.product_code, amount=args.amount, term_months=args.term_months
    )


def _knowledge_handler(retriever: Retriever | None) -> Handler:
    """search_banking_policy needs the retriever, so its handler is built per server."""

    async def search_banking_policy(
        session: AsyncSession, principal: Principal, args: SearchBankingPolicyInput, now: datetime
    ) -> BaseModel:
        if retriever is None:
            raise ServiceUnavailableError("The knowledge base is not available.")
        try:
            result = await retriever.retrieve(args.query, args.top_k)
        except EmbeddingError as exc:
            raise ServiceUnavailableError("The knowledge base is temporarily unavailable.") from exc
        return PolicySearchResult(
            query=result.query,
            passages=[
                PolicyPassage(
                    source=hit.chunk.source,
                    title=hit.chunk.title,
                    section=hit.chunk.section,
                    text=hit.chunk.text,
                    score=round(hit.score, 4),
                )
                for hit in result.passages
            ],
            sufficient=result.sufficient,
            reason=result.reason,
            embedding_model=retriever.embedder.name,
        )

    return search_banking_policy


HANDLERS: dict[str, Handler] = {
    "get_account_balance": _get_account_balance,
    "get_account_details": _get_account_details,
    "get_transactions": _get_transactions,
    "search_transactions": _search_transactions,
    "analyze_spending": _analyze_spending,
    "get_exchange_rate": _get_exchange_rate,
    "calculate_loan_payment": _calculate_loan_payment,
}


def _result(payload: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=is_error,
    )


def _error(code: str, message: str) -> CallToolResult:
    return _result({"error": {"code": code, "message": message}}, is_error=True)


async def _run_tool(
    spec: ToolSpec, handler: Handler, deps: ToolServerDeps, ctx: Context, arguments: dict[str, Any]
) -> CallToolResult:
    meta = ctx.request_context.meta or {}
    # Continue the caller's trace: our log lines carry the agent's request ID.
    trace = request_id_var.set(safe_request_id(meta.get(REQUEST_ID_META_KEY)))
    try:
        return await _execute_tool(spec, handler, deps, meta, arguments)
    finally:
        request_id_var.reset(trace)


async def _execute_tool(
    spec: ToolSpec, handler: Handler, deps: ToolServerDeps, meta: Mapping[str, Any], arguments: dict[str, Any]
) -> CallToolResult:
    started = time.perf_counter()
    log: dict[str, Any] = {"tool": spec.name}

    def finish(event: str, result: CallToolResult, **extra: Any) -> CallToolResult:
        log["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(event, extra={**log, **extra})
        return result

    try:
        user_id = verify_principal_token(meta.get(PRINCIPAL_META_KEY), secret=deps.auth_secret)
        log["user_id"] = str(user_id)
        args = spec.input_model.model_validate(arguments)
        async with deps.session_factory() as session:
            output = await handler(session, Principal(user_id), args, deps.clock())
    except InvalidTokenError:
        return finish(
            "mcp.tool.unauthenticated", _error("unauthenticated", "Missing or invalid principal token.")
        )
    except ValidationError as exc:
        message = describe_validation_error(exc)
        return finish("mcp.tool.invalid_arguments", _error("invalid_arguments", message))
    except DomainError as exc:
        return finish("mcp.tool.rejected", _error(exc.code, exc.message), code=exc.code)
    except (SQLAlchemyError, OSError):
        logger.exception("mcp.tool.backend_error", extra=log)
        return finish("mcp.tool.failed", _error("unavailable", "Banking data is temporarily unavailable."))
    except Exception:
        # A bug. Log everything for engineers; tell the caller nothing internal.
        logger.exception("mcp.tool.unexpected_error", extra=log)
        return finish("mcp.tool.failed", _error("internal_error", "The tool failed unexpectedly."))

    return finish("mcp.tool.completed", _result(output.model_dump(mode="json")))


def _make_tool_function(spec: ToolSpec, handler: Handler, deps: ToolServerDeps) -> Callable[..., Any]:
    """Build the function MCPServer registers, with a signature derived from the input model.

    MCPServer generates a tool's JSON schema from the Python signature. Instead
    of hand-writing seven signatures that duplicate the input models, we give
    one generic function a ``__signature__`` built from the model's fields
    (types, defaults, constraints, descriptions). The input model stays the
    single source of truth.
    """

    async def tool(ctx: Context, **arguments: Any) -> CallToolResult:
        return await _run_tool(spec, handler, deps, ctx, arguments)

    parameters = [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context)]
    parameters += [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=info.annotation, default=info)
        for name, info in spec.input_model.model_fields.items()
    ]
    # Annotated[CallToolResult, Model]: we build the result ourselves, but MCPServer
    # still publishes Model as the output schema and validates successful results.
    returns = Annotated[CallToolResult, spec.output_model]
    tool.__signature__ = inspect.Signature(parameters, return_annotation=returns)  # type: ignore[attr-defined]
    tool.__annotations__ = {"ctx": Context, "return": returns}
    tool.__name__ = tool.__qualname__ = spec.name
    return tool


def build_mcp_server(deps: ToolServerDeps) -> MCPServer:
    server = MCPServer(
        name=SERVER_NAME, title="BankAI banking tools (MOCK)", instructions=SERVER_INSTRUCTIONS
    )
    handlers = {**HANDLERS, "search_banking_policy": _knowledge_handler(deps.retriever)}
    for spec in TOOLS.values():
        server.add_tool(
            _make_tool_function(spec, handlers[spec.name], deps),
            name=spec.name,
            title=spec.title,
            description=spec.description,
            annotations=READ_ONLY,
        )

    # Plain HTTP endpoint for container health checks (outside the MCP protocol).
    server.custom_route("/health", methods=["GET"])(_health)
    return server


async def _health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "tools": len(TOOLS)})


def create_http_app(server: MCPServer, settings: Settings) -> Starlette:
    """Streamable-HTTP ASGI app.

    * stateless: every request stands alone, so replicas can sit behind a load balancer.
    * JSON responses instead of SSE streams: our tools return one result each.
    * Host-header allow-list: protects against DNS-rebinding attacks.
    """
    return server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=settings.mcp_allowed_hosts),
        host=settings.mcp_host,
    )


def run() -> None:
    """Entry point: ``python -m app.mcp``."""
    settings = get_settings()
    sessions = get_sessionmaker()
    deps = ToolServerDeps(
        session_factory=sessions,
        auth_secret=settings.require_mcp_auth_secret(),
        retriever=create_retriever(settings, sessions),
    )
    server = build_mcp_server(deps)
    configure_logging(settings.log_level, settings.log_format)  # after MCPServer's own logging setup
    logger.info(
        "mcp.startup", extra={"host": settings.mcp_host, "port": settings.mcp_port, "tools": list(TOOLS)}
    )
    uvicorn.run(
        create_http_app(server, settings), host=settings.mcp_host, port=settings.mcp_port, log_config=None
    )
