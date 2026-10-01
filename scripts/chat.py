"""Chat with BankAI in the terminal (MOCK data).

    python -m scripts.chat                                  # as CUST-0001, tools in-process
    python -m scripts.chat --customer CUST-0003 --debug     # another mock customer, show routing
    python -m scripts.chat --message "What's my balance?"   # one question, then exit
    python -m scripts.chat --mcp-url http://localhost:8001/mcp   # use a running MCP server

"Signing in" by customer number is a MOCK login for local demos only. The LLM
is whatever LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL configure (none = offline).
"""

import argparse
import asyncio
import secrets
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.assistant import AssistantReply, BankingAssistant, PreviousTurn
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine
from app.mcp.client import MCPToolGateway
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.models import User
from app.rag.factory import create_retriever
from app.rag.retriever import Retriever

EXIT_WORDS = {"exit", "quit", "bye", ":q"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with BankAI (MOCK data).")
    parser.add_argument("--customer", default="CUST-0001", help="mock customer number to act as")
    parser.add_argument("--mcp-url", default=None, help="MCP server URL (default: run the tools in-process)")
    parser.add_argument("--message", default=None, help="ask one question and exit")
    parser.add_argument("--debug", action="store_true", help="show intent, tool and timing for each reply")
    return parser.parse_args(argv)


def describe_llm(settings: Settings) -> str:
    if settings.llm_provider == "none":
        return "none (offline: rules + templates)"
    return f"{settings.llm_model} via {settings.llm_base_url or 'api.openai.com'}"


def print_reply(reply: AssistantReply, *, debug: bool) -> None:
    print(f"bankai> {reply.response}")
    if debug:
        print(
            f"        [intent={reply.intent} via {reply.intent_source} | tool={reply.tool} "
            f"args={reply.tool_arguments} | answer={reply.response_source} | {reply.safety_status} | "
            f"llm_calls={reply.llm_calls} | {reply.latency_ms:.0f} ms]"
        )
        if reply.errors:
            print(f"        [errors: {', '.join(reply.errors)}]")


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING", "console")  # --debug prints a summary line per reply instead

    engine = create_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            user = await session.scalar(select(User).where(User.customer_number == args.customer.upper()))
        if user is None:
            print(
                f"No mock customer {args.customer}. Seed the database: python -m scripts.seed_db",
                file=sys.stderr,
            )
            return 1

        retriever: Retriever | None = None  # in-process mode only; a remote server has its own
        if args.mcp_url:
            gateway = MCPToolGateway(
                args.mcp_url,
                auth_secret=settings.require_mcp_auth_secret(),
                timeout_seconds=settings.tool_timeout_seconds,
            )
        else:
            # In-process: server and client share this process, so a one-off secret is enough.
            secret = secrets.token_urlsafe(32)
            retriever = create_retriever(settings, sessions)
            server = build_mcp_server(
                ToolServerDeps(session_factory=sessions, auth_secret=secret, retriever=retriever)
            )
            gateway = MCPToolGateway(
                server, auth_secret=secret, timeout_seconds=settings.tool_timeout_seconds
            )
        bot = BankingAssistant.from_settings(settings, gateway=gateway)
        try:
            return await _converse(bot, user, args, describe_llm(settings))
        finally:
            await bot.aclose()
            if retriever is not None:
                await retriever.aclose()
    finally:
        await engine.dispose()


async def _converse(bot: BankingAssistant, user: User, args: argparse.Namespace, llm: str) -> int:
    if args.message is not None:
        print_reply(await bot.respond(user_id=user.id, message=args.message), debug=args.debug)
        return 0

    print(f"BankAI - MOCK data prototype. Signed in as {user.full_name} ({user.customer_number}).")
    print(f"LLM: {llm}. Type 'exit' to quit.")
    previous: PreviousTurn | None = None
    while True:
        try:
            line = await asyncio.to_thread(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in EXIT_WORDS:
            return 0
        if not line.strip():
            continue
        reply = await bot.respond(user_id=user.id, message=line, previous=previous)
        print_reply(reply, debug=args.debug)
        previous = reply.as_previous_turn() or previous


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
