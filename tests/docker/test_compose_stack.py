"""The real Docker stack: build the image, start all services, test them from the outside.

Marked ``docker``: skipped where Docker is not available, failed instead when
``REQUIRE_DOCKER=1`` (CI sets it, so a broken Docker setup can never pass silently).

The stack runs under its own Compose project name with free host ports, so it never
touches a stack you are running yourself, and it is removed (with its volumes) afterwards.
"""

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest

from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.docker
DOCKER = shutil.which("docker")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _docker_works() -> bool:
    if DOCKER is None:
        return False
    return subprocess.run([DOCKER, "info"], capture_output=True, check=False).returncode == 0  # noqa: S603


@dataclass
class Stack:
    project: str
    base: str
    env: dict[str, str]

    def compose(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        assert DOCKER is not None
        return subprocess.run(  # noqa: S603 - fixed arguments, our own compose file
            [DOCKER, "compose", "-p", self.project, *args],
            cwd=PROJECT_ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            check=check,
            timeout=600,
        )

    def exec(self, service: str, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.compose("exec", "-T", service, *command, check=check)

    def login(self, client: httpx.Client, customer: str = "CUST-0001") -> dict[str, str]:
        token = client.post("/auth/login", json={"customer_number": customer}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    if not _docker_works():
        if os.getenv("REQUIRE_DOCKER") == "1":
            pytest.fail("Docker is required (REQUIRE_DOCKER=1) but not available")
        pytest.skip("Docker is not available")
    backend_port = _free_port()
    env = {
        **os.environ,
        "BACKEND_PORT": str(backend_port),
        "POSTGRES_PORT": str(_free_port()),
    }
    for name in ("MCP_AUTH_SECRET", "API_SESSION_SECRET", "OPERATOR_API_KEY"):
        env.pop(name, None)  # the test environment's own secrets must not leak into the stack
    running = Stack(f"bankai-test-{uuid.uuid4().hex[:8]}", f"http://127.0.0.1:{backend_port}", env)
    try:
        up = running.compose("up", "--build", "--detach", "--wait", "--wait-timeout", "300", check=False)
        if up.returncode != 0:
            logs = running.compose("logs", "--no-color", "--tail", "40", check=False).stdout
            pytest.fail(f"docker compose up failed:\n{up.stderr[-3000:]}\n{logs[-4000:]}")
        yield running
    finally:
        running.compose("down", "--volumes", "--remove-orphans", check=False)


@pytest.fixture
def client(stack: Stack) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=stack.base, timeout=60) as http:
        yield http


def test_all_services_are_healthy_and_the_init_job_succeeded(stack: Stack, client: httpx.Client) -> None:
    states = {
        row["Service"]: (row["State"], row.get("Health", ""), row.get("ExitCode"))
        for row in map(json.loads, stack.compose("ps", "--all", "--format", "json").stdout.splitlines())
    }
    assert states["init"][0] == "exited" and states["init"][2] == 0
    assert all(states[s][:2] == ("running", "healthy") for s in ("database", "mcp-server", "backend"))
    health = client.get("/health").json()
    assert health["database"] == "ok" and health["components"]["tools"] == "mcp (http)"


def test_a_customer_is_answered_through_the_mcp_server(stack: Stack, client: httpx.Client) -> None:
    headers = stack.login(client)

    reply = client.post("/chat", json={"message": "What's my current balance?"}, headers=headers).json()
    policy = client.post(
        "/chat", json={"message": "What are the fees for international transfers?"}, headers=headers
    )

    assert reply["trace"]["tool"] == "get_account_balance" and "GEL" in reply["response"]
    assert "0.20% of the amount" in policy.json()["response"]  # knowledge base indexed in pgvector
    assert "MOCK" in client.get("/ui/").text


def test_another_customers_data_is_refused(stack: Stack, client: httpx.Client) -> None:
    giorgi = stack.exec(
        "database", "psql", "-U", "bankai", "-d", "bankai", "-tAc",
        "select id from users where customer_number = 'CUST-0002'",
    ).stdout.strip()  # fmt: skip
    response = client.get(f"/accounts/{giorgi}", headers=stack.login(client))
    assert response.status_code == 403


def test_the_mcp_server_and_the_database_are_not_open_to_the_network(stack: Stack) -> None:
    assert stack.compose("port", "mcp-server", "8001", check=False).stdout.strip() == ""
    database = stack.compose("port", "database", "5432").stdout.strip()
    assert database.startswith("127.0.0.1:")


def test_containers_run_unprivileged_on_a_read_only_file_system(stack: Stack) -> None:
    for service in ("backend", "mcp-server"):
        assert stack.exec(service, "id", "-u").stdout.strip() == "10001"
        assert stack.exec(service, "touch", "/app/written", check=False).returncode != 0
    listing = stack.exec("backend", "ls", "-A", "/app").stdout.split()
    assert ".env" not in listing and "tests" not in listing


def test_secrets_are_generated_private_and_work(stack: Stack, client: httpx.Client) -> None:
    modes = stack.exec("backend", "stat", "-c", "%a %U", *[f"/var/lib/bankai/secrets/{n}" for n in (
        "mcp_auth_secret", "api_session_secret", "operator_api_key")]).stdout.split("\n")  # fmt: skip
    assert all(line == "600 bankai" for line in filter(None, modes))
    key = stack.exec("backend", "cat", "/var/lib/bankai/secrets/operator_api_key").stdout.strip()

    assert client.get("/evaluations").status_code == 403
    started = client.post(
        "/evaluations/run", json={"categories": ["GENERAL"], "limit": 3}, headers={"X-Operator-Key": key}
    )
    assert started.status_code == 202  # (needs aiosqlite at run time - found by building this image)
    for _ in range(60):
        run = client.get(f"/evaluations/{started.json()['run_id']}", headers={"X-Operator-Key": key}).json()
        if run["status"] != "running":
            break
        time.sleep(1)
    assert run["status"] == "completed" and run["total_cases"] == 3


def test_running_the_init_job_again_changes_nothing(stack: Stack) -> None:
    def customers() -> str:
        return stack.exec(
            "database", "psql", "-U", "bankai", "-d", "bankai", "-tAc", "select count(*) from users"
        ).stdout.strip()

    secret_before = stack.exec("backend", "cat", "/var/lib/bankai/secrets/mcp_auth_secret").stdout
    again = stack.compose("run", "--rm", "--no-deps", "init")

    assert "seed.skipped" in again.stdout + again.stderr
    assert customers() == "10"
    assert stack.exec("backend", "cat", "/var/lib/bankai/secrets/mcp_auth_secret").stdout == secret_before
