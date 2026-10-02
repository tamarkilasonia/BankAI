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
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.assistant import AssistantReply, BankingAssistant, PendingConfirmation, PreviousTurn
from app.agents.factory import open_assistant
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine
from app.models import User

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
        if reply.safety_flags:
            print(f"        [safety: {', '.join(reply.safety_flags)}]")
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

        async with open_assistant(settings, sessions, mcp_url=args.mcp_url) as bot:
            return await _converse(bot, user, args, describe_llm(settings))
    finally:
        await engine.dispose()


async def _converse(bot: BankingAssistant, user: User, args: argparse.Namespace, llm: str) -> int:
    if args.message is not None:
        print_reply(await bot.respond(user_id=user.id, message=args.message), debug=args.debug)
        return 0

    print(f"BankAI - MOCK data prototype. Signed in as {user.full_name} ({user.customer_number}).")
    print(f"LLM: {llm}. Type 'exit' to quit.")
    previous: PreviousTurn | None = None
    pending: PendingConfirmation | None = None  # a high-risk request waiting for "yes" / "no"
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
        reply = await bot.respond(user_id=user.id, message=line, previous=previous, pending=pending)
        print_reply(reply, debug=args.debug)
        previous = reply.as_previous_turn() or previous
        pending = reply.pending_confirmation


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
