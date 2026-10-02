"""API test fixtures: the real app with the real agent over the seeded test bank.

Only the edges are replaced: the speech engines are fakes and the clock is pinned
to ANCHOR, so answers are deterministic.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.agents.assistant import BankingAssistant
from app.agents.nodes import AgentDeps
from app.api.conversations import ConversationStore
from app.api.main import create_app
from app.api.services import Services
from app.core.config import Settings
from app.mcp.client import MCPToolGateway
from tests.constants import ANCHOR, GIORGI, NINO, TEST_MCP_SECRET
from tests.fakes import FakeSTT, FakeTTS

SESSION_SECRET = "test-only-session-secret-0123456789abcdef"
OPERATOR_KEY = "test-only-operator-key-0123456789abcdef"


@dataclass
class Api:
    client: httpx.AsyncClient
    services: Services
    stt: FakeSTT
    tts: FakeTTS

    async def login(self, customer: str = "CUST-0001") -> dict[str, str]:
        response = await self.client.post("/auth/login", json={"customer_number": customer})
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    async def chat(self, headers: dict[str, str], message: str, conversation_id: str | None = None) -> Any:
        body = {"message": message} | ({"conversation_id": conversation_id} if conversation_id else {})
        response = await self.client.post("/chat", json=body, headers=headers)
        assert response.status_code == 200, response.text
        return response.json()


def api_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "operator_api_key": OPERATOR_KEY,
        "api_session_secret": SESSION_SECRET,
        "voice_max_audio_bytes": 200_000,
        "voice_max_audio_seconds": 30.0,
        "tts_max_chars": 500,
        "embedding_provider": "hashing",
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[call-arg]


@pytest.fixture
async def api(
    tool_server: MCPServer, seeded_engine: AsyncEngine, reset_app_state: None
) -> AsyncIterator[Api]:
    stt, tts = FakeSTT(), FakeTTS()
    gateway = MCPToolGateway(tool_server, auth_secret=TEST_MCP_SECRET)
    settings = api_settings()
    services = Services(
        settings=settings,
        sessions=async_sessionmaker(seeded_engine, expire_on_commit=False),
        assistant=BankingAssistant(AgentDeps(gateway=gateway, clock=lambda: ANCHOR)),
        session_secret=SESSION_SECRET,
        conversations=ConversationStore(timedelta(minutes=30)),
        stt=stt,  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
    )
    app = create_app(services)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bankai.test") as client:
            yield Api(client, services, stt, tts)
    await services.aclose()


@pytest.fixture
def nino_id() -> str:
    return str(NINO)


@pytest.fixture
def giorgi_id() -> str:
    return str(GIORGI)
