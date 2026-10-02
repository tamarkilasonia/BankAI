"""Wire a ready-to-use assistant: MCP gateway (+ in-process tool server and RAG) and LLM.

Used by the terminal chat, the safety suite and (later) the API and the evaluation
runner, so they all run exactly the same agent.
"""

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.assistant import BankingAssistant
from app.core.config import Settings
from app.mcp.client import MCPToolGateway
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.rag.factory import create_retriever
from app.rag.retriever import Retriever


@asynccontextmanager
async def open_assistant(
    settings: Settings, sessions: async_sessionmaker[AsyncSession], *, mcp_url: str | None = None
) -> AsyncIterator[BankingAssistant]:
    """An assistant whose tools run in-process (default) or on a remote MCP server (``mcp_url``)."""
    retriever: Retriever | None = None  # in-process mode only; a remote server has its own
    if mcp_url:
        secret = settings.require_mcp_auth_secret()
        gateway = MCPToolGateway(mcp_url, auth_secret=secret, timeout_seconds=settings.tool_timeout_seconds)
    else:
        # Server and client share this process, so a one-off secret is enough.
        secret = secrets.token_urlsafe(32)
        retriever = create_retriever(settings, sessions)
        server = build_mcp_server(
            ToolServerDeps(session_factory=sessions, auth_secret=secret, retriever=retriever)
        )
        gateway = MCPToolGateway(server, auth_secret=secret, timeout_seconds=settings.tool_timeout_seconds)
    bot = BankingAssistant.from_settings(settings, gateway=gateway, extra_secrets=[secret])
    try:
        yield bot
    finally:
        await bot.aclose()
        if retriever is not None:
            await retriever.aclose()
