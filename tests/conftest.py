"""Shared test fixtures.

Unit tests use an in-memory SQLite database: no server, no Docker, and every
test gets a brand-new empty schema. Behaviour that depends on PostgreSQL itself
(migrations, exact NUMERIC handling, SQL aggregates) is covered separately by
``tests/integration``.
"""

import logging
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

# Must happen before any application module reads settings: unit tests must
# never talk to a real database or a real LLM by accident.
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_API_KEY"] = ""
os.environ["MCP_AUTH_SECRET"] = "test-only-mcp-secret-0123456789abcdef"

import pytest
from hypothesis import settings as hypothesis_settings
from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  - registers all tables on Base.metadata
from app.core.config import get_settings
from app.database.base import Base
from app.database.seed import seed_database
from app.database.session import create_engine, get_engine, get_sessionmaker
from app.mcp.client import MCPToolGateway
from app.mcp.server import ToolServerDeps, build_mcp_server
from app.rag.documents import Chunk, load_knowledge_base, vocabulary
from app.rag.embeddings import HashingEmbedder
from app.rag.factory import DEFAULT_MIN_SCORES
from app.rag.indexing import index_chunks
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore
from app.services.authorization import Principal
from tests.constants import ANCHOR, GIORGI, NINO, TEST_MCP_SECRET

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Property-based tests (tests/property): no per-example time limit - CI machines vary.
# HYPOTHESIS_PROFILE=thorough explores 20x more examples (a nightly job, or after a rule change).
hypothesis_settings.register_profile("default", deadline=None)
hypothesis_settings.register_profile("thorough", deadline=None, max_examples=2000)
hypothesis_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Code under test may call configure_logging(); undo it after each test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


@pytest.fixture
async def reset_app_state() -> AsyncIterator[None]:
    """Clear cached settings/engine before and after a test that changes env vars."""
    get_settings.cache_clear()
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()
    yield
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
    get_settings.cache_clear()
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """A fresh in-memory SQLite database with the full schema."""
    # StaticPool: every session shares the one in-memory connection
    # (otherwise each new connection would see an empty database).
    engine = create_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session


@pytest.fixture
async def seeded_session(session: AsyncSession) -> AsyncSession:
    """A session on a database filled with the standard mock dataset."""
    await seed_database(session, anchor=ANCHOR)
    return session


@pytest.fixture
async def seeded_engine(engine: AsyncEngine) -> AsyncEngine:
    """An engine whose database holds the standard mock dataset."""
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        await seed_database(session, anchor=ANCHOR)
    return engine


@pytest.fixture(scope="session")
def knowledge_chunks() -> list[Chunk]:
    return load_knowledge_base(PROJECT_ROOT / "knowledge_base")


@pytest.fixture
async def retriever(knowledge_chunks: list[Chunk]) -> Retriever:
    """Retrieval over the real knowledge base with the offline hashing embedder."""
    embedder, store = HashingEmbedder(), InMemoryVectorStore()
    await index_chunks(knowledge_chunks, embedder, store)
    return Retriever(
        embedder=embedder,
        store=store,
        known_words=vocabulary(knowledge_chunks),
        min_score=DEFAULT_MIN_SCORES["hashing"],
    )


@pytest.fixture
def tool_server(seeded_engine: AsyncEngine, retriever: Retriever) -> MCPServer:
    """The real MCP server over the seeded database, with the clock pinned to ANCHOR."""
    deps = ToolServerDeps(
        session_factory=async_sessionmaker(seeded_engine, expire_on_commit=False),
        auth_secret=TEST_MCP_SECRET,
        clock=lambda: ANCHOR,
        retriever=retriever,
    )
    return build_mcp_server(deps)


@pytest.fixture
def gateway(tool_server: MCPServer) -> MCPToolGateway:
    """The agent-side gateway, connected to the server in-process (real MCP protocol, no network)."""
    return MCPToolGateway(tool_server, auth_secret=TEST_MCP_SECRET)


@pytest.fixture
def nino() -> Principal:
    return Principal(NINO)


@pytest.fixture
def giorgi() -> Principal:
    return Principal(GIORGI)
